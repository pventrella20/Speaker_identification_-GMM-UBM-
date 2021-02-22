from functools import reduce

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from scipy.io import wavfile
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture

from features_extraction import extract_features
from map_adaptation import map_adapt
from files_manager import file_processing

GMM_DATA_PATH = './data/gmm_dataset'
UBM_DATA_PATH = './data/ubm_dataset_1000'
TEST_DATA_PATH = './data/test'

SPEAKERS, ALL_SPEAKERS, TEST_SPEAKERS, SPEAKERS_NAMES = file_processing(GMM_DATA_PATH, UBM_DATA_PATH, TEST_DATA_PATH)

MODEL_SPEAKERS = len(SPEAKERS)
TOTAL_SPEAKERS = len(ALL_SPEAKERS)
TOTAL_TEST_SPEAKERS = len(TEST_SPEAKERS)
SPEAKERS_NUMBER = len(SPEAKERS_NAMES)

TRAIN_SPLITS = 7  # numero di segmenti usati per il training di ogni speaker

FEATURE_ORDER = 40  # numero di features
NUMBER_OF_GAUSSIAN = 512  # numero di componenti


def cmatrix_display(accuracy, confusion_matrix_spk, figsize=(10, 7)):
    """
    visualizza a video la matrice di confusione
    :param accuracy: accuracy della predizione
    :param confusion_matrix_spk: matrice di confusione
    :param figsize: grandezza dell'immagine
    :return:
    """
    df_cm = pd.DataFrame(confusion_matrix_spk, index=[i for i in SPEAKERS_NAMES],
                         columns=[i for i in SPEAKERS_NAMES])
    plt.figure(figsize=figsize)
    plt.title('Confusion Matrix\n Accuracy: {0:.3f}'.format(accuracy))
    sn.heatmap(df_cm, cmap='rocket_r', annot=True, fmt='g')
    plt.ylabel('True speaker')
    plt.xlabel('Predicted speaker')
    plt.show()


class SpeakerRecognition:

    def setGMMUBM(self, no_components):
        """
        crea lo spazio per calcolare le GMM e la UBM (verranno calcolate in fit_model)
        :param no_components: numero di distribuzioni gaussiane
        :return:
        """
        self.GMM = []
        self.UBM = []
        for i in range(MODEL_SPEAKERS):
            self.GMM.append(GaussianMixture(n_components=no_components, covariance_type='diag'))
            self.UBM.append(GaussianMixture(n_components=no_components, covariance_type='diag'))

    def load_data(self, fitted=True):
        """
        carica in memoria i dati relativi agli audio da processare
        :param fitted: True se il training è già stato effettuato, False altrimenti
        :return:
        """

        # carico in memoria ed estraggo le features dei file di test
        self.p_spk = [wavfile.read(TEST_DATA_PATH + '/' + k + '.wav') for k in TEST_SPEAKERS]
        for i in TEST_SPEAKERS:
            sr, audio = wavfile.read(TEST_DATA_PATH + '/' + i + '.wav')
            features = extract_features(audio, sr)
            self.p_spk_mfcc.append(features)
            print("features extracted for {}".format(i))

        # carico in memoria i file di addestramento delle GMM (necessari per associare un nome al relativo modello)
        self.spk = [wavfile.read(GMM_DATA_PATH + '/' + i + '.wav') for i in SPEAKERS]

        if fitted is False:
            # carico in memoria i file per l'addestramento della UBM
            self.all_spk = [wavfile.read(UBM_DATA_PATH + '/' + j + '.wav') for j in ALL_SPEAKERS]

            # estraggo le features dei file di addestramento delle GMM
            features = np.asarray(())
            j = 1
            for i in SPEAKERS:
                sr, audio = wavfile.read(GMM_DATA_PATH + '/' + i + '.wav')
                vector = extract_features(audio, sr)
                if features.size == 0:
                    features = vector
                else:
                    features = np.vstack((features, vector))
                if j == TRAIN_SPLITS:
                    self.spk_mfcc.append(features)
                    features = np.asarray(())
                    j = 0
                j += 1
                print("features extracted for {}".format(i))

            # estraggo le features dei file di addestramento della UBM
            for i in ALL_SPEAKERS:
                sr, audio = wavfile.read(UBM_DATA_PATH + '/' + i + '.wav')
                features = extract_features(audio, sr)
                self.all_spk_mfcc.append(features)
                print("features extracted for {}".format(i))
            self.cepstral_mean_subtraction(self.all_spk_mfcc)

            # unifico i file necessari per il training della UBM
            for i in range(TOTAL_SPEAKERS):
                self.spk_train_size.append(len(self.all_spk_mfcc[i]))
                self.spk_start.append(len(self.total_mfcc))
                print(i)
                for mfcc in self.all_spk_mfcc[i]:
                    self.total_mfcc.append(mfcc)
                    self.speaker_label.append(i)
                self.spk_end.append(len(self.total_mfcc))

    def fit_model(self):
        """
        addestra la UBM e la memorizza nella relativa list, dopodichè usa l'algoritmo MAP per modellare le GMM
        :return:
        """
        print("Fit start for UBM")
        self.UBM[0].fit(self.total_mfcc)
        joblib.dump(self.UBM[0], 'data/model/ubm' + '.pkl')
        print("Fit end for UBM")
        for i in range(SPEAKERS_NUMBER):
            print("Fit start for {}".format(SPEAKERS_NAMES[i]))
            gmm_means = map_adapt(self.UBM[0], self.spk_mfcc[i], NUMBER_OF_GAUSSIAN)
            self.GMM[i] = self.UBM[0]
            self.GMM[i].means_ = gmm_means
            joblib.dump(self.GMM[i], 'data/model/gmm' + str(i + 1) + "_" + SPEAKERS_NAMES[i] + '.pkl')
            print("Fit end for {}".format(SPEAKERS_NAMES[i]))

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
        for i in range(0, SPEAKERS_NUMBER):
            self.GMM.append(joblib.load('data/model/gmm' + str(i + 1) + "_" + SPEAKERS_NAMES[i] + '.pkl'))
        self.UBM.append(joblib.load('data/model/ubm' + '.pkl'))

    def predict(self):
        """
        effettua la predizione sui file di testing
        :return:
        """
        confusion_score = [[0 for _ in range(SPEAKERS_NUMBER)] for _ in range(TOTAL_TEST_SPEAKERS)]

        for i in range(TOTAL_TEST_SPEAKERS):
            for j in range(SPEAKERS_NUMBER):
                x = self.GMM[j].score_samples(self.p_spk_mfcc[i]) - self.UBM[0].score_samples(self.p_spk_mfcc[i])
                for score in x:
                    if score > 0:
                        confusion_score[i][j] += round(score, 2)
            print("Speaker evaluation {}/{} end".format(i + 1, TOTAL_TEST_SPEAKERS))

        for i in range(TOTAL_TEST_SPEAKERS):
            best_guess, _ = max(enumerate(confusion_score[i]), key=lambda p: p[1])

            print("For speaker {}, best guess is {}".format(TEST_SPEAKERS[i], SPEAKERS_NAMES[best_guess]))

            for speaker in SPEAKERS_NAMES:
                if speaker in TEST_SPEAKERS[i]:
                    self.y_true.append(speaker)
                    self.y_predict.append(SPEAKERS_NAMES[best_guess])

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

    def __init__(self):
        self.test_spk = []
        self.test_mfcc = []
        self.train_mfcc = []
        self.all_mfcc = []

        # Speaker data and corresponding mfcc
        self.spk = []
        self.spk_mfcc = []

        self.p_spk = []
        self.p_spk_mfcc = []

        self.all_spk = []
        self.all_spk_mfcc = []

        # Holds all the training mfccs of all speakers and
        # speaker_label is the speaker label for the corresponding mfcc

        self.total_mfcc = []
        self.speaker_label = []
        self.spk_train_size = []  # Index upto which is training data for that speaker.

        self.p_total_mfcc = []
        self.p_speaker_label = []
        self.spk_test_size = []

        # Since the length of all the audio files are different, spk_start and spk_end hold

        self.spk_start = []
        self.spk_end = []

        self.p_spk_start = []
        self.p_spk_end = []

        self.y_true = []
        self.y_predict = []

        self.GMM = []
        self.UBM = []
        self.load_data()
        self.cepstral_mean_subtraction(self.spk_mfcc)
        self.cepstral_mean_subtraction(self.p_spk_mfcc)


if __name__ == '__main__':
    """
    il risultato finale è una matrice di confusione che rende visuale l'accuratezza del sistema
    """
    SR = SpeakerRecognition()
    # SR.model_training(NUMBER_OF_GAUSSIAN)
    SR.load_model()
    confusion = SR.predict()
    print("Confusion Matrix")
    # noinspection PyDeprecation
    print(np.matrix(confusion))
    print("Accuracy score: {}".format(accuracy_score(SR.y_true, SR.y_predict)))
    print(classification_report(SR.y_true, SR.y_predict, target_names=SPEAKERS_NAMES))
    cm = confusion_matrix(SR.y_true, SR.y_predict, SPEAKERS_NAMES)
    cmatrix_display(accuracy_score(SR.y_true, SR.y_predict), cm)
