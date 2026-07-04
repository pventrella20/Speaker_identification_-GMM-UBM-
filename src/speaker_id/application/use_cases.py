"""Use-cases: orchestration layer for repositories, feature extraction and models.

Everything here is dependency-injected (paths, config, extractor, repositories,
classifier), so each use-case is unit-testable in isolation.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from ..audio.io import AudioRepository
from ..audio.splitter import AudioSplitter
from ..config import DataPaths, TrainingConfig
from ..domain.naming import parse_speaker_label, SpeakerFilenameError
from ..features.mfcc import MFCCFeatureExtractor, cepstral_mean_normalize
from ..ml.classifier import SpeakerClassifier, FrameVotingClassifier
from ..ml.gmm_ubm import adapt_speaker_model
from ..ml.persistence import ModelRepository

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]
UBM_FIT_STAGE_START = 20.0
UBM_FIT_STAGE_END = 35.0
UBM_MAX_EM_ITERATIONS = 100
UBM_PROGRESS_EMIT_EVERY = 2


def _emit_progress(on_progress: ProgressCallback | None, percent: float, message: str) -> None:
    if on_progress is not None:
        on_progress(max(0.0, min(100.0, percent)), message)


@dataclass
class SplitAudioUseCase:
    paths: DataPaths

    def execute(self, seconds_per_segment: int, on_progress: ProgressCallback | None = None) -> dict[str, list]:
        repo = AudioRepository(self.paths.temp)
        splitter = AudioSplitter(self.paths.temp, self.paths.splitted)
        stems = repo.list_stems()
        logger.info("Splitting %d file(s) from %s into %ds segments", len(stems), self.paths.temp, seconds_per_segment)

        if not stems:
            _emit_progress(on_progress, 100.0, "No files to split")
            return {}

        _emit_progress(on_progress, 0.0, f"Splitting {len(stems)} file(s)")
        output: dict[str, list] = {}
        total = len(stems)
        for idx, stem in enumerate(stems, start=1):
            output[stem] = splitter.split(stem, seconds_per_segment)
            percent = (idx / total) * 100.0
            _emit_progress(on_progress, percent, f"Split {idx}/{total}: {stem}")
        return output


@dataclass
class TrainModelsUseCase:
    paths: DataPaths
    config: TrainingConfig
    feature_extractor: MFCCFeatureExtractor
    model_repository: ModelRepository

    def execute(self, on_progress: ProgressCallback | None = None) -> list[str]:
        """Train UBM then MAP-adapt one speaker GMM per enrolled speaker."""
        ubm_repo = AudioRepository(self.paths.ubm_dataset)
        gmm_repo = AudioRepository(self.paths.gmm_dataset)

        ubm_stems = ubm_repo.list_stems()
        if not ubm_stems:
            raise FileNotFoundError(f"No .wav files found in {self.paths.ubm_dataset}")

        logger.info("Extracting UBM features from %d file(s)", len(ubm_stems))
        _emit_progress(on_progress, 0.0, "Preparing training data")

        def ubm_extract_progress(done: int, total: int, stem: str) -> None:
            percent = (done / total) * 15.0
            _emit_progress(on_progress, percent, f"UBM features {done}/{total}: {stem}")

        ubm_features = np.vstack(self._extract_all(ubm_repo, ubm_stems, on_item=ubm_extract_progress))
        ubm_features = cepstral_mean_normalize(ubm_features)
        _emit_progress(on_progress, 15.0, "UBM features ready")

        logger.info("Fitting UBM with %d Gaussian components", self.config.n_gaussians)
        ubm = self._fit_ubm_with_progress(ubm_features, on_progress)
        self.model_repository.save_ubm(ubm)
        _emit_progress(on_progress, UBM_FIT_STAGE_END, "UBM fitted and saved")

        speakers = self._group_by_speaker(gmm_repo.list_stems())
        if not speakers:
            raise FileNotFoundError(
                f"No correctly-named enrollment files found in {self.paths.gmm_dataset} "
                "(expected '#_name_surname_note.wav')"
            )

        speaker_labels = sorted(speakers)
        total_speakers = len(speaker_labels)

        for speaker_idx, label in enumerate(speaker_labels):
            stems = speakers[label]
            stage_start = UBM_FIT_STAGE_END + (65.0 * speaker_idx / total_speakers)
            stage_end = UBM_FIT_STAGE_END + (65.0 * (speaker_idx + 1) / total_speakers)
            extract_end = stage_start + (stage_end - stage_start) * 0.5

            logger.info("Adapting speaker model for '%s' (%d file(s))", label, len(stems))

            def speaker_extract_progress(done: int, total: int, stem: str) -> None:
                if total <= 0:
                    _emit_progress(on_progress, stage_start, f"Adapting {label}")
                    return
                ratio = done / total
                percent = stage_start + (extract_end - stage_start) * ratio
                _emit_progress(
                    on_progress,
                    percent,
                    f"Extracting features for {label} ({done}/{total}): {stem}",
                )

            features = np.vstack(self._extract_all(gmm_repo, stems, on_item=speaker_extract_progress))
            features = cepstral_mean_normalize(features)
            speaker_gmm = adapt_speaker_model(ubm, features, self.config.map_relevance_factor)
            self.model_repository.save_speaker_model(label, speaker_gmm)
            _emit_progress(on_progress, stage_end, f"Adapted {speaker_idx + 1}/{total_speakers}: {label}")

        logger.info("Training complete: %d speaker model(s) + 1 UBM", len(speakers))
        _emit_progress(on_progress, 100.0, "Training complete")
        return speaker_labels

    def _extract_all(
        self,
        repo: AudioRepository,
        stems: list[str],
        on_item: Callable[[int, int, str], None] | None = None,
    ) -> list[np.ndarray]:
        features = []
        total = len(stems)
        for idx, stem in enumerate(stems, start=1):
            clip = repo.load(stem)
            features.append(self.feature_extractor.extract(clip.samples, clip.sample_rate))
            if on_item is not None:
                on_item(idx, total, stem)
        return features

    def _fit_ubm_with_progress(
        self,
        ubm_features: np.ndarray,
        on_progress: ProgressCallback | None,
    ) -> GaussianMixture:
        """Fit UBM and emit per-iteration EM progress (throttled)."""
        ubm = GaussianMixture(
            n_components=self.config.n_gaussians,
            covariance_type="diag",
            max_iter=1,
            warm_start=True,
        )

        _emit_progress(on_progress, UBM_FIT_STAGE_START, f"Fitting UBM (EM 0/{UBM_MAX_EM_ITERATIONS})")

        converged = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            for em_iter in range(1, UBM_MAX_EM_ITERATIONS + 1):
                ubm.fit(ubm_features)
                converged = bool(getattr(ubm, "converged_", False))

                should_emit = (
                    (em_iter % UBM_PROGRESS_EMIT_EVERY == 0)
                    or converged
                    or em_iter == UBM_MAX_EM_ITERATIONS
                )
                if should_emit:
                    percent = UBM_FIT_STAGE_START + (
                        (em_iter / UBM_MAX_EM_ITERATIONS) * (UBM_FIT_STAGE_END - UBM_FIT_STAGE_START)
                    )
                    _emit_progress(on_progress, percent, f"Fitting UBM (EM {em_iter}/{UBM_MAX_EM_ITERATIONS})")

                if converged:
                    break

        if not converged:
            logger.warning(
                "UBM fitting did not converge after %d EM iterations; continuing with latest estimate.",
                UBM_MAX_EM_ITERATIONS,
            )
        return ubm

    @staticmethod
    def _group_by_speaker(stems: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for stem in stems:
            try:
                speaker = parse_speaker_label(stem)
            except SpeakerFilenameError as exc:
                logger.warning("Skipping '%s': %s", stem, exc)
                continue
            groups.setdefault(speaker.label, []).append(stem)
        return groups


@dataclass(frozen=True)
class IdentificationResult:
    file_stem: str
    predicted_label: str
    true_label: str | None
    scores: dict[str, float]


@dataclass
class IdentifySpeakersUseCase:
    paths: DataPaths
    feature_extractor: MFCCFeatureExtractor
    model_repository: ModelRepository
    classifier: SpeakerClassifier = field(default_factory=FrameVotingClassifier)

    def execute(self, on_progress: ProgressCallback | None = None) -> list[IdentificationResult]:
        test_repo = AudioRepository(self.paths.test)
        test_stems = test_repo.list_stems()
        if not test_stems:
            raise FileNotFoundError(f"No .wav files found in {self.paths.test}")

        ubm = self.model_repository.load_ubm()
        labels = self.model_repository.list_speaker_labels()
        if not labels:
            raise FileNotFoundError("No trained speaker models found. Run 'train' first.")
        speaker_models = {label: self.model_repository.load_speaker_model(label) for label in labels}

        _emit_progress(on_progress, 0.0, f"Identifying {len(test_stems)} file(s)")
        results: list[IdentificationResult] = []
        total = len(test_stems)
        for idx, stem in enumerate(test_stems, start=1):
            clip = test_repo.load(stem)
            features = self.feature_extractor.extract(clip.samples, clip.sample_rate)
            features = cepstral_mean_normalize(features)

            predicted_label, scores = self.classifier.classify(features, ubm, speaker_models)
            true_label = self._infer_true_label(stem, labels)
            logger.info("file=%s predicted=%s true=%s", stem, predicted_label, true_label)
            results.append(IdentificationResult(stem, predicted_label, true_label, scores))
            _emit_progress(on_progress, (idx / total) * 100.0, f"Processed {idx}/{total}: {stem}")
        return results

    @staticmethod
    def _infer_true_label(stem: str, labels: list[str]) -> str | None:
        for label in labels:
            if label in stem:
                return label
        return None

