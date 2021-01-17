from pydub import AudioSegment
import math


class SplitWavAudio:
    def __init__(self, folder1, filename):
        self.folder = folder1
        self.filename = filename
        self.filepath = folder1 + '/' + filename

        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1 * 1000
        t2 = to_sec * 1 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '/' + split_filename, format="wav")

    def multiple_split(self, sec_per_split):
        total_secs = math.ceil(self.get_duration())
        for i in range(0, total_secs, sec_per_split):
            split_fn = self.filename.replace('.wav', '') + '_' + str(int(i/sec_per_split)) + '.wav'
            self.single_split(i, i + sec_per_split, split_fn)
            print(str(int(i/sec_per_split)) + ' Done')
            if i == total_secs - sec_per_split:
                print('All splitted successfully')

if __name__ == '__main__':
    folder = './data/gmm_dataset'
    folder_test = './data/test'
    file_names = ['AA_alberto_angela', 'AC_andrea_camilleri', 'GM_gianni_morandi', 'JB_papa_francesco',
                  'MB_mike_bongiorno', 'MG_mariastella_gelmini', 'MG_mario_giordano', 'MR_matteo_renzi',
                  'SF_sabrina_ferilli', 'SB_silvio_berlusconi', 'Z1_rosario_fiorello', 'Z2_maurizio_crozza', 'Z3_paola_cortellesi', 'Z4_neri_marcore']
    file_names_imitations = ['01_alberto_angela_marcore', '02_mike_bongiorno_fiorello', '03_gianni_morandi_fiorello', '04_papa_francesco_crozza']
    for file in file_names:
        split_wav = SplitWavAudio(folder, file + '.wav')
        split_wav.multiple_split(sec_per_split=20)
