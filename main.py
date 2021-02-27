import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from speaker_identification import SpeakerRecognition
from files_manager import read_files
from pre_processing import splitting

GMM_DATA_PATH = './data/gmm_dataset'
UBM_DATA_PATH = './data/ubm_dataset_1000'
TEST_DATA_PATH = './data/test'
TEMP_DATA_PATH = './data/temp'


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


def preprocessing(fold, sec):
    files = read_files(fold)
    splitting(fold, files, sec)


if __name__ == '__main__':
    """
    il risultato finale è una matrice di confusione che rende visuale l'accuratezza del sistema
    """
    print("# Speakers Identification System #")
    print("> please insert training files in 'gmm_dataset' folder and testing files in 'test' folder")
    flag = True
    choice = ""
    choice2 = ""
    secs = 0
    ngauss = 0
    end_program = False
    while not end_program:
        flag = True
        while flag:
            choice = input("> write 'train' for training or 'test' for testing: ")
            if choice == "test" or choice == "train":
                flag = False
        flag = True
        while flag:
            choice2 = input("> do you need to split training or testing files? (yes/no): ")
            if choice2 == "yes" or choice2 == "no":
                flag = False
        if choice2 == "yes":
            print("> insert files to split into 'temp' folder (remember to execute 'convert.sh' first)")
            flag = True
            while flag:
                choice2 = input("> how many seconds per split? ")
                secs = int(choice2)
                if isinstance(secs, int):
                    flag = False
                print(isinstance(secs, int))
            preprocessing(TEMP_DATA_PATH, secs)
            print("> move MANUALLY training splits into 'gmm_dataset' folder and testing splits into 'test' folder")
            choice2 = input("> press ENTER when you're ready to start training/testing...")

        if choice == "train":
            flag = True
            while flag:
                choice2 = input("> insert number of Gaussian components for models (a power of 2): ")
                ngauss = int(choice2)
                if math.log(ngauss, 2).is_integer():
                    flag = False
            SR = SpeakerRecognition(GMM_DATA_PATH, UBM_DATA_PATH, TEST_DATA_PATH, ngauss, False)
            SR.model_training()
            print("> training complete!")
        else:
            SR = SpeakerRecognition(GMM_DATA_PATH, UBM_DATA_PATH, TEST_DATA_PATH)
            SR.load_model()
            confusion, confusion_p = SR.predict()
            print("> scoring complete!")
            print("Score Confusion Matrix")
            # noinspection PyDeprecation
            print(np.matrix(confusion))
            print("Percentage Confusion Matrix")
            # noinspection PyDeprecation
            print(np.matrix(confusion_p))
            print("Accuracy score: {}".format(accuracy_score(SR.y_true, SR.y_predict)))
            print(classification_report(SR.y_true, SR.y_predict, target_names=SR.speakers_names))
            cm = confusion_matrix(SR.y_true, SR.y_predict, SR.speakers_names)
            cmatrix_display(accuracy_score(SR.y_true, SR.y_predict), cm, SR.speakers_names)
            end_program = True
