import numpy as np
import copy

FEATURES_NUMBER = 40
SCALING_FACTOR = 0.01


def map_adapt(ubm, features, NUMBER_OF_GAUSSIAN):
    probability = ubm.predict_proba(features)
    n_i = np.sum(probability, axis=0)

    E = np.zeros((FEATURES_NUMBER, NUMBER_OF_GAUSSIAN), dtype=np.float32)
    for ii in range(0, NUMBER_OF_GAUSSIAN):
        probability_gauss = np.tile(probability[:, ii], (FEATURES_NUMBER, 1)).T * features
        if n_i[ii] == 0:
            E[:, ii] = 0
        else:
            E[:, ii] = np.sum(probability_gauss, axis=0) / n_i[ii]

    alpha = n_i / (n_i + SCALING_FACTOR)

    old_mean = copy.deepcopy(ubm.means_)
    new_mean = np.zeros((NUMBER_OF_GAUSSIAN, FEATURES_NUMBER), dtype=np.float32)

    for ii in range(0, NUMBER_OF_GAUSSIAN):
        new_mean[ii, :] = (alpha[ii] * E[:, ii]) + ((1 - alpha[ii]) * old_mean[ii, :])

    return new_mean
