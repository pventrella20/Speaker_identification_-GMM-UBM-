from functools import reduce

import joblib
import numpy as np
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture

from features_extraction import extract_features
from map_adaptation import map_adapt
from files_manager import file_processing

NUMBER_OF_GAUSSIAN = 512  # numero di componenti gaussiane


class SpeakerRecognition:

    def setGMMUBM(self, no_components):
        """
        crea lo spazio per calcolare le GMM e la UBM (verranno calcolate in fit_model)
        :param no_components: numero di distribuzioni gaussiane
        :return:
        """
        self.GMM = []
        self.UBM = []
        for i in range(self.gmm_files):
            self.GMM.append(GaussianMixture(n_components=no_components, covariance_type='diag'))
        self.UBM.append(GaussianMixture(n_components=no_components, covariance_type='diag'))

    def load_data(self, fitted=True):
        """
        carica in memoria i dati relativi agli audio da processare
        :param fitted: True se il training è già stato effettuato, False altrimenti
        :return:
        """

        # carico in memoria ed estraggo le features dei file di test
        self.p_spk = [wavfile.read(self.test_data_path + '/' + k + '.wav') for k in self.test_files]
        for i in self.test_files:
            sr, audio = wavfile.read(self.test_data_path + '/' + i + '.wav')
            features = extract_features(audio, sr)
            self.p_spk_mfcc.append(features)
            print("features extracted for {}".format(i))

        # carico in memoria i file di addestramento delle GMM (necessari per associare un nome al relativo modello)
        self.spk = [wavfile.read(self.gmm_data_path + '/' + i + '.wav') for i in self.gmm_files]

        if fitted is False:
            # carico in memoria i file per l'addestramento della UBM
            self.all_spk = [wavfile.read(self.ubm_data_path + '/' + j + '.wav') for j in self.ubm_files]

            # estraggo le features dei file di addestramento delle GMM
            features = np.asarray(())
            j = 1
            for i in self.gmm_files:
                sr, audio = wavfile.read(self.gmm_data_path + '/' + i + '.wav')
                vector = extract_features(audio, sr)
                if features.size == 0:
                    features = vector
                else:
                    features = np.vstack((features, vector))
                if j == self.train_splits:
                    self.spk_mfcc.append(features)
                    features = np.asarray(())
                    j = 0
                j += 1
                print("features extracted for {}".format(i))

            # estraggo le features dei file di addestramento della UBM
            for i in self.ubm_files:
                sr, audio = wavfile.read(self.ubm_data_path + '/' + i + '.wav')
                features = extract_features(audio, sr)
                self.all_spk_mfcc.append(features)
                print("features extracted for {}".format(i))
            self.cepstral_mean_subtraction(self.all_spk_mfcc)

            # unifico i file necessari per il training della UBM
            for i in range(self.ubm_speakers_n):
                for mfcc in self.all_spk_mfcc[i]:
                    self.total_mfcc.append(mfcc)

    def fit_model(self):
        """
        addestra la UBM e la memorizza nella relativa list, dopodichè usa l'algoritmo MAP per modellare le GMM
        :return:
        """
        print("Fit start for UBM")
        self.UBM[0].fit(self.total_mfcc)
        joblib.dump(self.UBM[0], 'data/model/ubm' + '.pkl')
        print("Fit end for UBM")
        for i in range(self.speakers_number):
            print("Fit start for {}".format(self.speakers_names[i]))
            gmm_means = map_adapt(self.UBM[0], self.spk_mfcc[i], NUMBER_OF_GAUSSIAN)
            self.GMM[i] = self.UBM[0]
            self.GMM[i].means_ = gmm_means
            joblib.dump(self.GMM[i], 'data/model/gmm' + str(i + 1) + "_" + self.speakers_names[i] + '.pkl')
            print("Fit end for {}".format(self.speakers_names[i]))

    def model_training(self, no_components=NUMBER_OF_GAUSSIAN):
        """
        alloca lo spazio e addestra i modelli
        :param no_components: numero di distribuzioni gaussiane
        :return:
        """
        self.setGMMUBM(no_components)
        self.fit_model()

    def load_model(self):
        """
        carica in memoria (a partire dai file relativi) i modelli precedentemente addestrati e memorizzati
        :return:
        """
        for i in range(0, self.speakers_number):
            self.GMM.append(joblib.load('data/model/gmm' + str(i + 1) + "_" + self.speakers_names[i] + '.pkl'))
        self.UBM.append(joblib.load('data/model/ubm' + '.pkl'))

    def predict(self):
        """
        effettua la predizione sui file di testing
        :return:
        """
        confusion_score = [[0 for _ in range(self.speakers_number)] for _ in range(self.test_speakers_n)]

        for i in range(self.test_speakers_n):
            for j in range(self.speakers_number):
                x = self.GMM[j].score_samples(self.p_spk_mfcc[i]) - self.UBM[0].score_samples(self.p_spk_mfcc[i])
                for score in x:
                    if score > 0:
                        confusion_score[i][j] += round(score, 2)
            print("Speaker evaluation {}/{} end".format(i + 1, self.test_speakers_n))

        for i in range(self.test_speakers_n):
            best_guess, _ = max(enumerate(confusion_score[i]), key=lambda p: p[1])

            print("For speaker {}, best guess is {}".format(self.test_files[i], self.speakers_names[best_guess]))

            for speaker in self.speakers_names:
                if speaker in self.test_files[i]:
                    self.y_true.append(speaker)
                    self.y_predict.append(self.speakers_names[best_guess])

        return confusion_score

    @staticmethod
    def cepstral_mean_subtraction(mfcc_vector):
        """
        normalizzazione delle features (migliora le performance)
        :param mfcc_vector: vettore di features
        :return:
        """
        for i, speaker_mfcc in enumerate(mfcc_vector):
            average = reduce(lambda acc, ele: acc + ele, speaker_mfcc)
            average = list(map(lambda x: x / len(speaker_mfcc), average))
            for j, feature_vector in enumerate(speaker_mfcc):
                for k, feature in enumerate(feature_vector):
                    mfcc_vector[i][j][k] -= average[k]

    def __init__(self, gmmpath, ubmpath, testpath, splits, fitted=True):
        # dati degli speakers e relativi MFCC
        self.spk = []
        self.spk_mfcc = []

        self.p_spk = []
        self.p_spk_mfcc = []

        self.all_spk = []
        self.all_spk_mfcc = []

        # lista unificata di tutti gli MFCC degli speaker della UBM
        self.total_mfcc = []

        # vettori dei valori attuali e predetti (speaker reale - speaker predetto)
        self.y_true = []
        self.y_predict = []

        # lista dei modelli
        self.GMM = []
        self.UBM = []

        # path dei dati da processare
        self.gmm_data_path = gmmpath
        self.test_data_path = testpath
        self.ubm_data_path = ubmpath

        # lista dei file da processare
        self.gmm_files = []
        self.test_files = []
        self.ubm_files = []
        # nome degli speakers del modello
        self.speakers_names = []

        # avvaloro le liste di ogni categoria di file audio a partire dai rispettivi path
        self.gmm_files, self.ubm_files, self.test_files, self.speakers_names = \
            file_processing(self.gmm_data_path, self.ubm_data_path, self.test_data_path)

        # numero di file da processare
        self.gmm_speakers_n = len(self.gmm_files)
        self.test_speakers_n = len(self.test_files)
        self.ubm_speakers_n = len(self.ubm_files)
        # numero degli speakers del modello
        self.speakers_number = len(self.speakers_names)

        # numero di segmenti audio per l'addestramento
        self.train_splits = splits

        # caricamento dei dati
        self.load_data(fitted)
        # normalizzazione delle features
        self.cepstral_mean_subtraction(self.spk_mfcc)
        self.cepstral_mean_subtraction(self.p_spk_mfcc)
