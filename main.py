import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from speaker_identification import SpeakerRecognition

GMM_DATA_PATH = './data/gmm_dataset'
UBM_DATA_PATH = './data/ubm_dataset_1000'
TEST_DATA_PATH = './data/test'
TRAIN_SPLITS = 7  # numero di audio per l'addestramento di ogni speaker


def cmatrix_display(accuracy, confusion_matrix_spk, spk_names, figsize=(10, 7)):
    """
    visualizza a video la matrice di confusione
    :param spk_names: lista dei parlatori per cui il modello è addestrato
    :param accuracy: accuracy della predizione
    :param confusion_matrix_spk: matrice di confusione
    :param figsize: grandezza dell'immagine
    :return:
    """
    df_cm = pd.DataFrame(confusion_matrix_spk, index=[i for i in spk_names],
                         columns=[i for i in spk_names])
    plt.figure(figsize=figsize)
    plt.title('Confusion Matrix\n Accuracy: {0:.3f}'.format(accuracy))
    sn.heatmap(df_cm, cmap='rocket_r', annot=True, fmt='g')
    plt.ylabel('True speaker')
    plt.xlabel('Predicted speaker')
    plt.show()


if __name__ == '__main__':
    """
    il risultato finale è una matrice di confusione che rende visuale l'accuratezza del sistema
    """
    SR = SpeakerRecognition(GMM_DATA_PATH, UBM_DATA_PATH, TEST_DATA_PATH, TRAIN_SPLITS)
    # SR.model_training(NUMBER_OF_GAUSSIAN)
    SR.load_model()
    confusion = SR.predict()
    print("Confusion Matrix")
    # noinspection PyDeprecation
    print(np.matrix(confusion))
    print("Accuracy score: {}".format(accuracy_score(SR.y_true, SR.y_predict)))
    print(classification_report(SR.y_true, SR.y_predict, target_names=SR.speakers_names))
    cm = confusion_matrix(SR.y_true, SR.y_predict, SR.speakers_names)
    cmatrix_display(accuracy_score(SR.y_true, SR.y_predict), cm, SR.speakers_names)
