import python_speech_features as psf
from sklearn.mixture import GaussianMixture
import joblib
from scipy.io import wavfile
from functools import reduce
import numpy as np
import soundfile as sf

from os import listdir
from os.path import isfile, join
import re

GMM_DATA_PATH = './data/gmm_dataset'
UBM_DATA_PATH = './data/ubm_dataset'
TEST_DATA_PATH = './data/test'

ubm_files = [f for f in listdir(UBM_DATA_PATH) if isfile(join(UBM_DATA_PATH, f))]
ubm_files.sort()
ubm_speakers = []
for filename in ubm_files:
    dups = re.search('[\w].mp3', filename)
    if dups is None and filename != 'convert.sh':
        ubm_speakers.append(''.join(filename.split('.wav')[0]))
print(ubm_speakers)

gmm_files = [f for f in listdir(GMM_DATA_PATH) if isfile(join(GMM_DATA_PATH, f))]
gmm_files.sort()
gmm_speakers = []
for filename in gmm_files:
    dups = re.search('[\w].mp3', filename)
    if dups is None and filename != 'convert.sh':
        gmm_speakers.append(''.join(filename.split('.wav')[0]))
print(gmm_speakers)

test_files = [f for f in listdir(TEST_DATA_PATH) if isfile(join(TEST_DATA_PATH, f))]
test_files.sort()
test_speakers = []
for filename in test_files:
    dups = re.search('[\w].mp3', filename)
    if dups is None and filename != 'convert.sh':
        test_speakers.append(''.join(filename.split('.wav')[0]))
print(test_speakers)

SPEAKERS = gmm_speakers
ALL_SPEAKERS = ubm_speakers
TEST_SPEAKERS = test_speakers

MODEL_SPEAKERS = len(SPEAKERS)
TOTAL_SPEAKERS = len(ALL_SPEAKERS)
TOTAL_TEST_SPEAKERS = len(TEST_SPEAKERS)

class SpeakerRecognition:

    #  Create a GMM and UBM model for each speaker. The GMM is modelled after the speaker and UBM for each speaker
    #  is modelled after all the other speakers. Likelihood Ratio test is used to verify speaker
    def setGMMUBM(self, no_components):
        self.GMM = []
        self.UBM = []
        for i in range(MODEL_SPEAKERS):
            self.GMM.append(GaussianMixture(n_components=no_components, covariance_type='diag'))
            self.UBM.append(GaussianMixture(n_components=no_components, covariance_type='diag'))

    # Load in data from .wav files in data/
    # Extract mfcc (first 13 coefficients) from each audio sample
    def load_data(self):

        self.spk = [wavfile.read(GMM_DATA_PATH + '/' + i + '.wav') for i in SPEAKERS]
        self.spk_mfcc = [psf.mfcc(self.spk[i][1], self.spk[i][0]) for i in range(0, MODEL_SPEAKERS)]

        self.all_spk = [wavfile.read(UBM_DATA_PATH + '/' + j + '.wav') for j in ALL_SPEAKERS]
        self.all_spk_mfcc = [psf.mfcc(self.all_spk[j][1], self.all_spk[j][0]) for j in range(0, TOTAL_SPEAKERS)]

        self.train_mfcc += self.spk_mfcc
        self.train_mfcc += self.all_spk_mfcc

        self.p_spk = [wavfile.read(TEST_DATA_PATH + '/' + k + '.wav') for k in TEST_SPEAKERS]
        self.p_spk_mfcc = [psf.mfcc(self.p_spk[k][1], self.p_spk[k][0]) for k in range(0, TOTAL_TEST_SPEAKERS)]

        self.test_mfcc += self.p_spk_mfcc
        self.test_mfcc += self.all_spk_mfcc

        for i in range(MODEL_SPEAKERS + TOTAL_SPEAKERS):
            self.spk_train_size.append(len(self.train_mfcc[i]))
            self.spk_start.append(len(self.total_mfcc))
            print(i)
            for mfcc in self.train_mfcc[i]:
                self.total_mfcc.append(mfcc)
                self.speaker_label.append(i)
            self.spk_end.append(len(self.total_mfcc))

        for i in range(MODEL_SPEAKERS + TOTAL_SPEAKERS):
            self.spk_test_size.append(len(self.test_mfcc[i]))
            self.spk_start.append(len(self.p_total_mfcc))
            print(i)
            for mfcc in self.test_mfcc[i]:
                self.p_total_mfcc.append(mfcc)
                self.p_speaker_label.append(i)
            self.p_spk_end.append(len(self.p_total_mfcc))


    # Gaussian Mixture Model is made of a number of Gaussian distribution components.
    # To model data, a suitable number o gaussian components have to be selected.
    # There is no method for finding this. It is done by trial and error. This runs
    # the program for different values of component and records accuracy for each one
    def find_best_params(self):
        best_no_components = 1
        maxacc = 0
        for i in range(100, 256):
            self.setGMMUBM(i)
            self.fit_model()
            _, acc, _ = self.predict()
            print("Accuracy for n = {} is {}".format(i, acc))
            if acc > maxacc:
                maxacc = acc
                best_no_components = i
        return best_no_components

    # Fit the GMM UBM models with training data
    def fit_model(self):
        for i in range(MODEL_SPEAKERS):
            print("Fit start for {}".format(i))
            self.GMM[i].fit(self.spk_mfcc[i])
            self.UBM[i].fit(self.total_mfcc[:self.spk_start[i]] + self.total_mfcc[self.spk_end[i]:])
            print("Fit end for {}".format(i))
            joblib.dump(self.UBM[i], 'data/model/ubm' + str(i) + '.pkl')
            joblib.dump(self.GMM[i], 'data/model/gmm' + str(i) + '.pkl')

    def model(self, no_components=244):
        self.setGMMUBM(no_components)
        self.fit_model()

    # Predict the output for each model for each speaker and produce confusion matrix
    def load_model(self):
        for i in range(0, MODEL_SPEAKERS):
            self.GMM.append(joblib.load('data/model/gmm' + str(i) + '.pkl'))
            self.UBM.append(joblib.load('data/model/ubm' + str(i) + '.pkl'))

    def predict(self):
        avg_accuracy = 0

        confusion = [[0 for y in range(MODEL_SPEAKERS)] for x in range(TOTAL_TEST_SPEAKERS)]

        for i in range(TOTAL_TEST_SPEAKERS):
            for j in range(MODEL_SPEAKERS):
                x = self.GMM[j].score_samples(self.p_spk_mfcc[i]) - self.UBM[j].score_samples(self.p_spk_mfcc[i])
                for score in x:
                    if score > 0:
                        confusion[i][j] += 1

        confusion_diag = [confusion[i][i] for i in range(MODEL_SPEAKERS)]

        diag_sum = 0
        for item in confusion_diag:
            diag_sum += item

        remain_sum = 0
        for i in range(MODEL_SPEAKERS):
            for j in range(MODEL_SPEAKERS):
                if i != j:
                    remain_sum += confusion[i][j]

        spk_accuracy = 0
        for i in range(TOTAL_TEST_SPEAKERS):
            best_guess, _ = max(enumerate(confusion[i]), key=lambda p: p[1])
            raw_values = []
            for j in range(MODEL_SPEAKERS):
                raw_values.append(confusion[i][j])
            percentage_accuracy = (max(raw_values)/sum(raw_values))*100
            print("For speaker {}, best guess is {} [{}% accuracy]".format(TEST_SPEAKERS[i], SPEAKERS[best_guess], round(percentage_accuracy, 2)))
            if TEST_SPEAKERS[i].__contains__(SPEAKERS[best_guess]):
                spk_accuracy += 1
        spk_accuracy /= TOTAL_TEST_SPEAKERS

        avg_accuracy = diag_sum/(remain_sum+diag_sum)
        return confusion, round(avg_accuracy, 2), round(spk_accuracy, 2)

    def __init__(self):
        self.test_spk = []
        self.test_mfcc = []
        self.train_mfcc = []

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

        self.GMM = []
        self.UBM = []
        self.load_data()
        self.cepstral_mean_subtraction()

    # Cepstral Mean Subtraction (Feature Normalization step)
    def cepstral_mean_subtraction(self):
        for i, speaker_mfcc in enumerate(self.spk_mfcc):
            average = reduce(lambda acc, ele: acc + ele, speaker_mfcc)
            average = list(map(lambda x: x/len(speaker_mfcc), average))
            for j, feature_vector in enumerate(speaker_mfcc):
                for k, feature in enumerate(feature_vector):
                    self.spk_mfcc[i][j][k] -= average[k]
        for i, speaker_mfcc in enumerate(self.p_spk_mfcc):
            average = reduce(lambda acc, ele: acc + ele, speaker_mfcc)
            average = list(map(lambda x: x / len(speaker_mfcc), average))
            for j, feature_vector in enumerate(speaker_mfcc):
                for k, feature in enumerate(feature_vector):
                    self.p_spk_mfcc[i][j][k] -= average[k]


#TBD : Ten fold validation
def ten_fold(self):
    fold_size = 0.1 * self.n
    fold_offset = 0.0
    accuracy_per_fold = 0
    average_accuracy = 0

    for i in range(0, 10):
        print("Fold start is {}  and fold end is {} ".format( fold_offset, fold_offset + fold_size))
        accuracy = self.execute(int(fold_offset), int(fold_offset + fold_size))
        print("Accuracy is of test {} is : {} ".format(i, accuracy))
        average_accuracy += accuracy
        fold_offset += fold_size

    average_accuracy /= 10.0
    print("Average accuracy  " + str(100 * average_accuracy))
    return average_accuracy


# Final result is a confusion matrix which represents the accuracy of the fit of the model
if __name__ == '__main__':

    SR = SpeakerRecognition()
    SR.load_model()
    #SR.setGMMUBM(no_components=13)
    #SR.find_best_params()
    #SR.fit_model()
    confusion, mfcc_accuracy, spk_accuracy = SR.predict()

    print("Confusion Matrix")
    print(np.matrix(confusion))
    print("Accuracy in predicting speakers : {}".format(spk_accuracy))
    print("Accuracy in testing for MFCC : {}".format(mfcc_accuracy))


