"""Proves the vectorized delta computation matches the legacy
`features_extraction.calculate_delta` on random inputs."""
import numpy as np
from speaker_id.features.mfcc import MFCCFeatureExtractor, cepstral_mean_normalize


def legacy_calculate_delta(array):
    """Literal re-implementation of the original `features_extraction.calculate_delta`."""
    rows, cols = array.shape
    deltas = np.zeros((rows, cols))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            first = 0 if i - j < 0 else i - j
            second = rows - 1 if i + j > rows - 1 else i + j
            index.append((second, first))
            j += 1
        deltas[i] = (
            array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))
        ) / 10
    return deltas


def test_vectorized_delta_matches_legacy_loop():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(50, 20))

    expected = legacy_calculate_delta(features)
    actual = MFCCFeatureExtractor._delta(features)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_cepstral_mean_normalize_zero_mean():
    rng = np.random.default_rng(1)
    features = rng.normal(loc=5.0, scale=2.0, size=(30, 10))
    normalized = cepstral_mean_normalize(features)
    np.testing.assert_allclose(normalized.mean(axis=0), np.zeros(10), atol=1e-10)


def test_cepstral_mean_normalize_does_not_mutate_input():
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    original = features.copy()
    cepstral_mean_normalize(features)
    np.testing.assert_array_equal(features, original)
