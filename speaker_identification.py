from functools import reduce

import joblib
import numpy as np
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture

from features_extraction import extract_features
from map_adaptation import map_adapt
from files_manager import file_processing
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


class SpeakerRecognition:

    def setGMMUBM(self, no_components):
        """
        crea lo spazio per calcolare le GMM e la UBM (verranno calcolate in fit_model)
        :param no_components: numero di distribuzioni gaussiane
        :return:
        """
        self.GMM = []
        self.UBM = []
        for i in range(self.gmm_speakers_n):
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
        print(">> starting test files feature extraction...")
        for i in self.test_files:
            sr, audio = wavfile.read(self.test_data_path + '/' + i + '.wav')
            features = extract_features(audio, sr)
            self.p_spk_mfcc.append(features)
            print("features extracted for {}".format(i))
        print(">> test files feature extraction complete!\n")

        # carico in memoria i file di addestramento delle GMM (necessari per associare un nome al relativo modello)
        self.spk = [wavfile.read(self.gmm_data_path + '/' + i + '.wav') for i in self.gmm_files]

        if fitted is False:
            # carico in memoria i file per l'addestramento della UBM
            self.all_spk = [wavfile.read(self.ubm_data_path + '/' + j + '.wav') for j in self.ubm_files]

            print(">> starting training files feature extraction...")
            # estraggo le features dei file di addestramento delle GMM
            features = np.asarray(())
            for name in self.speakers_names:
                for i in self.gmm_files:
                    if name in i:
                        sr, audio = wavfile.read(self.gmm_data_path + '/' + i + '.wav')
                        vector = extract_features(audio, sr)
                        if features.size == 0:
                            features = vector
                        else:
                            features = np.vstack((features, vector))
                self.spk_mfcc.append(features)
                features = np.asarray(())
                print("features extracted for {}".format(name))
            print(">> training files feature extraction complete!\n")

            # estraggo le features dei file di addestramento della UBM
            print(">> starting UBM files feature extraction...")
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
            print(">> UBM files feature extraction complete!\n")

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
            gmm_means = map_adapt(self.UBM[0], self.spk_mfcc[i], self.n_gaussian)
            self.GMM[i] = self.UBM[0]
            self.GMM[i].means_ = gmm_means
            joblib.dump(self.GMM[i], 'data/model/gmm' + str(i + 1) + "_" + self.speakers_names[i] + '.pkl')
            print("Fit end for {}".format(self.speakers_names[i]))

    def model_training(self):
        """
        alloca lo spazio e addestra i modelli
        """
        self.setGMMUBM(self.n_gaussian)
        self.fit_model()

    def load_model(self):
        """
        carica in memoria (a partire dai file relativi) i modelli precedentemente addestrati e memorizzati
        :return:
        """
        for i in range(0, self.speakers_number):
            self.GMM.append(joblib.load('data/model/gmm' + str(i + 1) + "_" + self.speakers_names[i] + '.pkl'))
        self.GMM.append(joblib.load('data/model/ubm' + '.pkl'))
        self.UBM.append(joblib.load('data/model/ubm' + '.pkl'))

    def predict(self):
        """
        effettua la predizione sui file di testing
        :return:
        """
        confusion_score = [[0 for _ in range(self.speakers_number+1)] for _ in range(self.test_speakers_n)]
        confusion_percent = [[0 for _ in range(self.speakers_number+1)] for _ in range(self.test_speakers_n)]

        ubm_index = []
        for i in range(self.test_speakers_n):
            ubm = True
            for j in range(self.speakers_number):
                x = self.GMM[j].score_samples(self.p_spk_mfcc[i]) - self.UBM[0].score_samples(self.p_spk_mfcc[i])
                for score in x:
                    if score > 0:
                         confusion_score[i][j] += round(score, 2)
                y = self.GMM[j].score(self.p_spk_mfcc[i]) - self.UBM[0].score(self.p_spk_mfcc[i])
                if y > 0:
                    ubm = False
                #confusion_score[i][j] += round(x, 2)
            print("speaker evaluation {}/{} end".format(i + 1, self.test_speakers_n))  # scoring
            if ubm:
                ubm_index.append(True)
            else:
                ubm_index.append(False)

        for i in range(self.test_speakers_n):  # restituisce il best_guess per lo speaker i
            if ubm_index[i] is True:
                best_guess = self.speakers_number
            else:
                best_guess, _ = max(enumerate(confusion_score[i]), key=lambda p: p[1])

            print("for speaker {}, best guess is {}".format(self.test_files[i], self.speakers_names[best_guess]))

            for speaker in self.speakers_names:  # crea i vettori (speaker reale - predizione)
                if speaker in self.test_files[i] and speaker != "ubm":
                    self.y_true.append(speaker)
                    self.y_predict.append(self.speakers_names[best_guess])

        # converto i nomi degli speakers in label (solo cognomi) per un miglior displaying nella matrice di confusione
        for i in range(self.speakers_number):
            for j in range(self.test_speakers_n):
                if self.speakers_label[i] in self.y_true[j]:
                    self.y_true[j] = self.speakers_label[i]
                if self.speakers_label[i] in self.y_predict[j]:
                    self.y_predict[j] = self.speakers_label[i]

        for i in range(self.test_speakers_n):
            row_values = sum(confusion_score[i])
            for j in range(self.speakers_number):
                if row_values == 0:
                    confusion_percent[i][j] += 0
                else:
                    confusion_percent[i][j] += round((confusion_score[i][j]/row_values)*100, 2)

        return confusion_score, confusion_percent

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

    def __init__(self, gmmpath, ubmpath, testpath, n_gauss=512, fitted=True):
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
        self.speakers_label = []

        # avvaloro le liste di ogni categoria di file audio a partire dai rispettivi path
        self.gmm_files, self.ubm_files, self.test_files, self.speakers_names, self.speakers_label = \
            file_processing(self.gmm_data_path, self.ubm_data_path, self.test_data_path)

        # numero di file da processare
        self.gmm_speakers_n = len(self.gmm_files)
        self.test_speakers_n = len(self.test_files)
        self.ubm_speakers_n = len(self.ubm_files)
        # numero degli speakers del modello
        self.speakers_number = len(self.speakers_names)-1

        # numero di componenti gaussiane
        self.n_gaussian = n_gauss

        # caricamento dei dati
        self.load_data(fitted)
        # normalizzazione delle features
        self.cepstral_mean_subtraction(self.spk_mfcc)
        self.cepstral_mean_subtraction(self.p_spk_mfcc)
