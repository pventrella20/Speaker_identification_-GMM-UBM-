import matplotlib.pyplot as plt
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
UBM_DATA_PATH = './data/ubm_dataset'
TEST_DATA_PATH = './data/test'
TEMP_DATA_PATH = './data/temp'
SPLIT_DATA_PATH = './data/splitted'


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
    plt.subplots_adjust(bottom=0.155, right=0.924)
    plt.show()


def preprocessing(fold1, fold2, sec):
    """
    divide i file nella cartella in segmenti da x secondi
    :param fold1: percorso della cartella
    :param fold2: percorso dei file splittati
    :param sec: secondi per segmento
    """
    files = read_files(fold1)
    splitting(fold1, fold2, files, sec)


def test():
    """
    effettua il testing dopo aver caricato in memoria i modelli memorizzati
    """
    print("> testing started...")
    SR = SpeakerRecognition(GMM_DATA_PATH, UBM_DATA_PATH, TEST_DATA_PATH)
    SR.load_model()
    SR.predict()
    print("> ...testing complete!")
    print("Accuracy score: {}\n".format(accuracy_score(SR.y_true, SR.y_predict)))
    #print(classification_report(SR.y_true, SR.y_predict, target_names=SR.speakers_label, zero_division=0))
    cm = confusion_matrix(SR.y_true, SR.y_predict, SR.speakers_label)
    cmatrix_display(accuracy_score(SR.y_true, SR.y_predict), cm, SR.speakers_label)


def train():
    """
    effettua il training dei modelli a partire dai file nella cartella 'gmm_dataset' e 'ubm_dataset'
    """
    print("> training started...")
    SR = SpeakerRecognition(GMM_DATA_PATH, UBM_DATA_PATH, TEST_DATA_PATH, ngauss, False)
    SR.model_training()
    print("> ...training complete!")


if __name__ == '__main__':
    """
    il risultato finale è una matrice di confusione che rende visuale l'accuratezza del sistema
    """
    print("### Speakers Identification System ###")
    flag = True
    choice = ""
    choice2 = ""
    secs = 0
    ngauss = 0
    end_program = False
    while not end_program:  # START PROGRAM
        print(">> REMINDER: rename every training file in this way -> '#_name_surname_notes'")
        print("  >> where '#' is a number that specify the order for speakers files displaying")
        print("  >> please insert training files in 'gmm_dataset' folder and testing files in 'test' folder")
        flag = True
        while flag:
            choice = input("> write 'train' for training or 'test' for testing: ")
            if choice == "test" or choice == "train":
                flag = False
        flag = True
        while flag:  # SPLITTING FILES
            choice2 = input("> do you need to split training or testing files? (yes/no): ")
            if choice2 == "yes" or choice2 == "no":
                flag = False
        if choice2 == "yes":
            print(">> insert files to split into 'temp' folder (remember to execute 'convert.sh' first)")
            flag = True
            while flag:
                choice2 = input("> how many seconds per split? ")
                secs = int(choice2)
                if isinstance(secs, int):
                    flag = False
            preprocessing(TEMP_DATA_PATH, SPLIT_DATA_PATH, secs)  # preprocessing dei file
            print(">> you can found splitted files into 'splitted' folder")
            choice2 = input("> write something when you're ready to start training/testing...")

        # FASE DI TRAINING
        if choice == "train":
            flag = True
            while flag:
                choice2 = input("> insert number of Gaussian components for models (a power of 2): ")
                ngauss = int(choice2)
                if math.log(ngauss, 2).is_integer():
                    flag = False
            train()  # start training
        # FASE DI TESTING
        else:
            test()  # start testing
            end_program = True
