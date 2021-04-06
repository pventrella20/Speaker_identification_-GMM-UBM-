import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc


def calculate_delta(array):
    """
    calcola le delta MFCC dell'array in input (20 dimensioni)
    :param array: array di features
    :return:
    """

    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas


def extract_features(audio, rate):
    """
    estrae i vettori MFCC a 20 dimensioni dal file audio e li unisce alle MFCC-delta a 20 dimensioni,
    creando un vettore di features a 40 dimensioni
    :param audio: file audio in input
    :param rate: frequenza del file audio
    :return:
    """

    mfcc_feat = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, appendEnergy=True)

    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat, delta))
    return combined
