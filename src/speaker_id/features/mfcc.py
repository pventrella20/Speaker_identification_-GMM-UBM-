"""MFCC + delta feature extraction.

Two numerical routines were rewritten from nested Python loops to vectorized
NumPy, with a unit test (tests/test_features.py) asserting bit-for-bit
equivalence against the original formulas on random inputs:

1. `_delta` replaces `features_extraction.calculate_delta`, which built the
   result one row and one lag at a time in pure Python.
2. `cepstral_mean_normalize` replaces
   `speaker_identification.SpeakerRecognition.cepstral_mean_subtraction`, which
   computed the column-wise mean with a manual `functools.reduce` and then
   subtracted it with a triple-nested loop over frames x coefficients x speakers.
   That version also mutated its input list of arrays in place, an easy source
   of hard-to-trace bugs if the same features are ever normalized twice.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn import preprocessing
import python_speech_features as psf


@dataclass(frozen=True)
class MFCCFeatureExtractor:
    n_mfcc: int = 20
    frame_size_sec: float = 0.025
    frame_step_sec: float = 0.01
    delta_lag: int = 2

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """40-dim feature vector per frame: 20 scaled MFCCs + 20 delta-MFCCs."""
        mfcc_feat = psf.mfcc(
            audio, sample_rate, self.frame_size_sec, self.frame_step_sec, self.n_mfcc, appendEnergy=True
        )
        mfcc_feat = preprocessing.scale(mfcc_feat)
        delta = self._delta(mfcc_feat, self.delta_lag)
        return np.hstack((mfcc_feat, delta))

    @staticmethod
    def _delta(features: np.ndarray, n: int = 2) -> np.ndarray:
        """Vectorized equivalent of the legacy `calculate_delta`.

        delta[i] = (f[i+1] - f[i-1] + 2*(f[i+2] - f[i-2])) / 10, with indices
        clamped to [0, rows-1] at the boundaries -- reproduced here via
        edge-padding, which is exactly what the original min/max clamping did.
        """
        rows = features.shape[0]
        padded = np.pad(features, ((n, n), (0, 0)), mode="edge")
        idx = np.arange(rows)
        plus1, minus1 = padded[idx + n + 1], padded[idx + n - 1]
        plus2, minus2 = padded[idx + n + 2], padded[idx + n - 2]
        return (plus1 - minus1 + 2 * (plus2 - minus2)) / 10.0


def cepstral_mean_normalize(feature_matrix: np.ndarray) -> np.ndarray:
    """Subtracts the per-column mean; returns a new array (does not mutate input)."""
    if feature_matrix.size == 0:
        return feature_matrix
    return feature_matrix - feature_matrix.mean(axis=0, keepdims=True)
