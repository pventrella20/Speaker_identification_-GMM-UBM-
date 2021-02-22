from pydub import AudioSegment
import math


class SplitWavAudio:
    """
    classe per lo split di un file audio
    """
    def __init__(self, folder1, filename):
        self.folder = folder1
        self.filename = filename
        self.filepath = folder1 + '/' + filename

        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        """
        ritorna la durata del file audio
        :return: durata dell'audio
        """
        return self.audio.duration_seconds

    def single_split(self, from_sec, to_sec, split_filename):
        """
        split singolo di un file audio
        :param from_sec: secondo di inizio taglio
        :param to_sec: secondo di fine taglio
        :param split_filename: nome del file
        :return:
        """
        t1 = from_sec * 1 * 1000
        t2 = to_sec * 1 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '/' + split_filename, format="wav")

    def multiple_split(self, sec_per_split):
        """
        split multiplo di un file audio
        :param sec_per_split: lunghezza di ogni split in secondi
        :return:
        """
        total_secs = math.ceil(self.get_duration())
        for i in range(0, total_secs, sec_per_split):
            split_fn = self.filename.replace('.wav', '') + '_' + str(int(i / sec_per_split)) + '.wav'
            self.single_split(i, i + sec_per_split, split_fn)
            print(str(int(i / sec_per_split)) + ' Done')
            if i == total_secs - sec_per_split:
                print('All splitted successfully')


def splitting(folder, file_names, sec_per_split):
    """
    splitta tutti i file in una cartella in audio della stessa lunghezza (in secondi)
    :param folder: cartella dei files
    :param file_names: nomi dei files
    :param sec_per_split: secondi per ogni split
    :return:
    """
    for file in file_names:
        split_wav = SplitWavAudio(folder, file + '.wav')
        split_wav.multiple_split(sec_per_split)
