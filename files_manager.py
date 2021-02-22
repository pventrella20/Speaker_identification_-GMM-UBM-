import re
from os import listdir
from os.path import isfile, join


def file_processing(GMM_DATA_PATH, UBM_DATA_PATH, TEST_DATA_PATH):
    """
    carica in memoria i nomi dei file
    :param GMM_DATA_PATH:
    :param UBM_DATA_PATH:
    :param TEST_DATA_PATH:
    :return:
    """

    # cerca e memorizza i nomi dei file audio per l'addestramento della UBM
    ubm_files = [f for f in listdir(UBM_DATA_PATH) if isfile(join(UBM_DATA_PATH, f))]
    ubm_files.sort()
    ubm_speakers = []
    for filename in ubm_files:
        dups = re.search('[\w].mp3', filename)
        if dups is None and filename != 'convert.sh':
            ubm_speakers.append(''.join(filename.split('.wav')[0]))
    print(ubm_speakers)

    # cerca e memorizza i nomi dei file audio (split) per l'addestramento delle GMM, unificando i nomi degli speakers
    gmm_files = [f for f in listdir(GMM_DATA_PATH) if isfile(join(GMM_DATA_PATH, f))]
    gmm_files.sort()
    gmm_speakers = []
    speakers = []
    for filename in gmm_files:
        dups = re.search('[\w].ogg', filename)
        if dups is None and filename != 'convert.sh':
            gmm_speakers.append(''.join(filename.split('.wav')[0]))
    for elem in gmm_speakers:
        curr_elem = elem.split("_")
        speakers.append(curr_elem[1] + "_" + curr_elem[2])
    speakers_names = list(dict.fromkeys(speakers))
    print(gmm_speakers)
    print(speakers_names)

    # cerca e memorizza i nomi dei file audio per il testing dei modelli
    test_files = [f for f in listdir(TEST_DATA_PATH) if isfile(join(TEST_DATA_PATH, f))]
    test_files.sort()
    test_speakers = []
    for filename in test_files:
        dups = re.search('[\w].mp3', filename)
        dups2 = re.search('[\w].ogg', filename)
        if dups is None and dups2 is None and filename != 'convert.sh':
            test_speakers.append(''.join(filename.split('.wav')[0]))
    print(test_speakers)

    return gmm_speakers, ubm_speakers, test_speakers, speakers_names