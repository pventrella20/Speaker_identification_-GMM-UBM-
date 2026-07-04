"""Speaker classification as a pluggable Strategy.

`FrameVotingClassifier` intentionally reproduces the exact decision rule from
`speaker_identification.SpeakerRecognition.predict`, just vectorized and
extracted from a 40-line method entangled with confusion-matrix bookkeeping:

  - for every frame, accumulate the *positive part* of the per-frame
    log-likelihood-ratio margin GMM_i(frame) - UBM(frame), for every enrolled
    speaker i (frames where a speaker scores below the UBM don't count against
    them, they just don't contribute);
  - the predicted speaker is the one with the highest accumulated margin;
  - if no enrolled speaker's utterance-level average log-likelihood exceeds the
    UBM's, the utterance is classified as unknown ("ubm").

This is now a `Protocol`-based Strategy: alternative decision rules (e.g. a pure
utterance-level likelihood ratio, or a softmax over frame scores) can be added
as new classes without touching `IdentifySpeakersUseCase` or the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np
from sklearn.mixture import GaussianMixture

UNKNOWN_LABEL = "ubm"


class SpeakerClassifier(Protocol):
    def classify(
        self,
        features: np.ndarray,
        ubm: GaussianMixture,
        speaker_models: dict[str, GaussianMixture],
    ) -> tuple[str, dict[str, float]]:
        """Returns (predicted_label, per-speaker score)."""
        ...


@dataclass
class FrameVotingClassifier:
    def classify(
        self,
        features: np.ndarray,
        ubm: GaussianMixture,
        speaker_models: dict[str, GaussianMixture],
    ) -> tuple[str, dict[str, float]]:
        if not speaker_models:
            return UNKNOWN_LABEL, {}

        ubm_frame_scores = ubm.score_samples(features)
        ubm_utterance_score = ubm.score(features)

        margins: dict[str, float] = {}
        any_speaker_above_ubm = False
        for label, gmm in speaker_models.items():
            frame_margins = gmm.score_samples(features) - ubm_frame_scores
            margins[label] = float(np.clip(frame_margins, a_min=0, a_max=None).sum())
            if gmm.score(features) > ubm_utterance_score:
                any_speaker_above_ubm = True

        if not any_speaker_above_ubm:
            return UNKNOWN_LABEL, margins
        best_label = max(margins, key=margins.get)
        return best_label, margins
