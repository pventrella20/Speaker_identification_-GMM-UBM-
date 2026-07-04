"""Use-cases: the only place that orchestrates repositories, feature extraction
and models together. Everything here is constructed via dependency injection
(paths, config, extractor, repositories, classifier are all passed in), so each
use-case can be unit-tested with fakes/mocks instead of real audio files and a
real filesystem -- the original `SpeakerRecognition` class did I/O, feature
extraction, training, persistence and prediction all in one object, which made
any of that impossible to test in isolation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
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


@dataclass
class SplitAudioUseCase:
    paths: DataPaths

    def execute(self, seconds_per_segment: int) -> dict[str, list]:
        repo = AudioRepository(self.paths.temp)
        splitter = AudioSplitter(self.paths.temp, self.paths.splitted)
        stems = repo.list_stems()
        logger.info("Splitting %d file(s) from %s into %ds segments", len(stems), self.paths.temp, seconds_per_segment)
        return splitter.split_all(stems, seconds_per_segment)


@dataclass
class TrainModelsUseCase:
    paths: DataPaths
    config: TrainingConfig
    feature_extractor: MFCCFeatureExtractor
    model_repository: ModelRepository

    def execute(self) -> list[str]:
        """Trains the UBM on `ubm_dataset`, then MAP-adapts one GMM per speaker
        found in `gmm_dataset`. Returns the list of trained speaker labels."""
        ubm_repo = AudioRepository(self.paths.ubm_dataset)
        gmm_repo = AudioRepository(self.paths.gmm_dataset)

        ubm_stems = ubm_repo.list_stems()
        if not ubm_stems:
            raise FileNotFoundError(f"No .wav files found in {self.paths.ubm_dataset}")
        logger.info("Extracting UBM features from %d file(s)", len(ubm_stems))
        ubm_features = np.vstack(self._extract_all(ubm_repo, ubm_stems))
        ubm_features = cepstral_mean_normalize(ubm_features)

        logger.info("Fitting UBM with %d Gaussian components", self.config.n_gaussians)
        ubm = GaussianMixture(n_components=self.config.n_gaussians, covariance_type="diag")
        ubm.fit(ubm_features)
        self.model_repository.save_ubm(ubm)

        speakers = self._group_by_speaker(gmm_repo.list_stems())
        if not speakers:
            raise FileNotFoundError(
                f"No correctly-named enrollment files found in {self.paths.gmm_dataset} "
                "(expected '#_name_surname_note.wav')"
            )

        for label, stems in speakers.items():
            logger.info("Adapting speaker model for '%s' (%d file(s))", label, len(stems))
            features = np.vstack(self._extract_all(gmm_repo, stems))
            features = cepstral_mean_normalize(features)
            speaker_gmm = adapt_speaker_model(ubm, features, self.config.map_relevance_factor)
            self.model_repository.save_speaker_model(label, speaker_gmm)

        logger.info("Training complete: %d speaker model(s) + 1 UBM", len(speakers))
        return sorted(speakers)

    def _extract_all(self, repo: AudioRepository, stems: list[str]) -> list[np.ndarray]:
        features = []
        for stem in stems:
            clip = repo.load(stem)
            features.append(self.feature_extractor.extract(clip.samples, clip.sample_rate))
        return features

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

    def execute(self) -> list[IdentificationResult]:
        test_repo = AudioRepository(self.paths.test)
        test_stems = test_repo.list_stems()
        if not test_stems:
            raise FileNotFoundError(f"No .wav files found in {self.paths.test}")

        ubm = self.model_repository.load_ubm()
        labels = self.model_repository.list_speaker_labels()
        if not labels:
            raise FileNotFoundError("No trained speaker models found. Run 'train' first.")
        speaker_models = {label: self.model_repository.load_speaker_model(label) for label in labels}

        results = []
        for stem in test_stems:
            clip = test_repo.load(stem)
            features = self.feature_extractor.extract(clip.samples, clip.sample_rate)
            features = cepstral_mean_normalize(features)

            predicted_label, scores = self.classifier.classify(features, ubm, speaker_models)
            true_label = self._infer_true_label(stem, labels)
            logger.info("file=%s predicted=%s true=%s", stem, predicted_label, true_label)
            results.append(IdentificationResult(stem, predicted_label, true_label, scores))
        return results

    @staticmethod
    def _infer_true_label(stem: str, labels: list[str]) -> str | None:
        for label in labels:
            if label in stem:
                return label
        return None
