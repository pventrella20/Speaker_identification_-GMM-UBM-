"""MAP adaptation: derive a per-speaker GMM from the shared UBM.

Two independent problems in the legacy implementation are fixed here:

1. Performance: `map_adaptation.map_adapt` looped over every Gaussian component
   in pure Python, calling `np.tile` per component. `map_adapt_means` below
   replaces the whole loop with a single matrix multiplication
   (posterior.T @ features), which is both faster and easier to verify
   against the mathematical definition of the sufficient statistic E_i.
   tests/test_map_adaptation.py checks numerical equivalence against a
   literal re-implementation of the original loop, on random data.

2. Correctness/aliasing: in `speaker_identification.SpeakerRecognition.fit_model`,
   each adapted model was built as:

       self.GMM[i] = self.UBM[0]
       self.GMM[i].means_ = gmm_means

   `self.GMM[i] = self.UBM[0]` does not copy the UBM -- it makes `self.GMM[i]`
   and `self.UBM[0]` two names for the *same* GaussianMixture object. Every
   iteration of the loop then mutates that one shared object's `.means_` in
   place. Because each speaker model is immediately serialized with
   `joblib.dump` inside the loop, the bug was invisible on disk (each dump
   captured a correct snapshot) -- but any in-memory use of `self.GMM` after
   training (before reloading from disk) would silently see every speaker
   model collapse to whichever speaker was adapted last, since they all point
   to the same object. `adapt_speaker_model` uses `copy.deepcopy` so every
   speaker model is a genuinely independent object, in memory and on disk.
"""
from __future__ import annotations

import copy
import numpy as np
from sklearn.mixture import GaussianMixture


def map_adapt_means(
    ubm_means: np.ndarray, posterior: np.ndarray, features: np.ndarray, relevance_factor: float
) -> np.ndarray:
    """Vectorized MAP adaptation of component means.

    :param ubm_means: (n_components, n_features) UBM means.
    :param posterior: (n_frames, n_components) responsibilities, i.e. ubm.predict_proba(features).
    :param features: (n_frames, n_features) speaker enrollment features.
    :param relevance_factor: regularization term r; alpha_i = n_i / (n_i + r).
    """
    n_i = posterior.sum(axis=0)  # (n_components,)
    weighted_sum = posterior.T @ features  # (n_components, n_features), the sufficient statistic
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.divide(
            weighted_sum, n_i[:, None], out=np.zeros_like(weighted_sum), where=n_i[:, None] != 0
        )
    alpha = n_i / (n_i + relevance_factor)
    return alpha[:, None] * expected + (1 - alpha[:, None]) * ubm_means


def adapt_speaker_model(
    ubm: GaussianMixture, features: np.ndarray, relevance_factor: float
) -> GaussianMixture:
    """Returns a new, independent GMM: a deep copy of `ubm` with MAP-adapted means."""
    posterior = ubm.predict_proba(features)
    new_means = map_adapt_means(ubm.means_, posterior, features, relevance_factor)
    adapted = copy.deepcopy(ubm)
    adapted.means_ = new_means
    return adapted
