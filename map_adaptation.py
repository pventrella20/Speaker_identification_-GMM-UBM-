import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import main


class Train_GMM:
    def returnGMM(self, features, NUMBER_OF_GAUSSIAN):
        gmm = GMM(n_components=NUMBER_OF_GAUSSIAN, covariance_type='diag')
        gmm.fit(features)
        return gmm

class opt:
    def normalize_meanvector(weight, var, mean_vec):
        normalize_mean = np.zeros(np.shape(mean_vec), dtype=np.float32)
        [NUMBER_OF_GAUSSIAN, FEATURE_ORDER] = np.shape(mean_vec)
        for ii in range(0, NUMBER_OF_GAUSSIAN):
            normalize_mean[ii, :] = np.sqrt(weight[ii]) * \
                            (1 / np.sqrt(var[ii, :])) * mean_vec[ii, :]
        return normalize_mean

NUMBER_OF_GAUSSIAN = 512
NUMBER_OF_SAMPLES = 500

# samples

