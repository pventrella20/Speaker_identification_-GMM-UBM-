"""Proves the vectorized MAP adaptation is numerically equivalent to the
original nested-loop implementation from the legacy `map_adaptation.py`."""
import numpy as np
from speaker_id.ml.gmm_ubm import map_adapt_means


def legacy_map_adapt(ubm_means, posterior, features, relevance_factor):
    """Literal re-implementation of the original `map_adaptation.map_adapt`."""
    n_features = features.shape[1]
    n_components = posterior.shape[1]
    n_i = np.sum(posterior, axis=0)
    E = np.zeros((n_features, n_components), dtype=np.float64)
    for ii in range(n_components):
        probability_gauss = np.tile(posterior[:, ii], (n_features, 1)).T * features
        if n_i[ii] == 0:
            E[:, ii] = 0
        else:
            E[:, ii] = np.sum(probability_gauss, axis=0) / n_i[ii]
    alpha = n_i / (n_i + relevance_factor)
    new_mean = np.zeros((n_components, n_features), dtype=np.float64)
    for ii in range(n_components):
        new_mean[ii, :] = (alpha[ii] * E[:, ii]) + ((1 - alpha[ii]) * ubm_means[ii, :])
    return new_mean


def test_vectorized_matches_legacy_loop():
    rng = np.random.default_rng(42)
    n_frames, n_components, n_features = 120, 8, 40

    ubm_means = rng.normal(size=(n_components, n_features))
    raw_posterior = rng.random(size=(n_frames, n_components))
    posterior = raw_posterior / raw_posterior.sum(axis=1, keepdims=True)
    features = rng.normal(size=(n_frames, n_features))

    expected = legacy_map_adapt(ubm_means, posterior, features, relevance_factor=0.01)
    actual = map_adapt_means(ubm_means, posterior, features, relevance_factor=0.01)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_handles_zero_responsibility_component():
    # A component with zero total posterior must fall back to the UBM mean (E=0 branch).
    ubm_means = np.array([[1.0, 2.0], [3.0, 4.0]])
    posterior = np.array([[1.0, 0.0], [1.0, 0.0]])  # component 1 never responsible
    features = np.array([[10.0, 20.0], [30.0, 40.0]])

    result = map_adapt_means(ubm_means, posterior, features, relevance_factor=0.01)
    np.testing.assert_allclose(result[1], ubm_means[1])
