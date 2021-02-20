import numpy as np
import copy
from sklearn.mixture import GaussianMixture as GMM


FEATURE_ORDER = 40
NUMBER_OF_GAUSSIAN = 512
SCALING_FACTOR = 0.01

class Map:
    def map_adaptation(self, ubm, features):
        probability = ubm.predict_proba(features)
        n_i = np.sum(probability, axis=0)

        E = np.zeros((FEATURE_ORDER, NUMBER_OF_GAUSSIAN), dtype=np.float32)
        for ii in range(0, NUMBER_OF_GAUSSIAN):
            probability_gauss = np.tile(probability[:, ii], (FEATURE_ORDER, 1)).T * features
            E[:, ii] = np.sum(probability_gauss, axis=0) / n_i[ii]

        alpha = n_i / (n_i + SCALING_FACTOR)

        old_mean = copy.deepcopy(ubm.means_)
        new_mean = np.zeros((NUMBER_OF_GAUSSIAN, FEATURE_ORDER), dtype=np.float32)

        for ii in range(0, NUMBER_OF_GAUSSIAN):
            new_mean[ii, :] = (alpha[ii] * E[:, ii]) + ((1 - alpha[ii]) * old_mean[ii, :])

        # normalize
        weight = ubm.weights_
        var = ubm.covariances_

        ubm.means_ = self.normalize_meanvector(weight, var, new_mean)
        return ubm

    def normalize_meanvector(self, weight, var, mean_vec):
        normalize_mean = np.zeros(np.shape(mean_vec), dtype=np.float32)
        [NUMBER_OF_GAUSSIAN, FEATURE_ORDER] = np.shape(mean_vec)
        for ii in range(0, NUMBER_OF_GAUSSIAN):
            normalize_mean[ii, :] = np.sqrt(weight[ii]) * \
                                    (1 / np.sqrt(var[ii, :])) * mean_vec[ii, :]
        return normalize_mean

    def returnGMM(self, features, NUMBER_OF_GAUSSIAN):
        gmm = GMM(n_components=NUMBER_OF_GAUSSIAN, covariance_type='diag')
        gmm.fit(features)
        return gmm