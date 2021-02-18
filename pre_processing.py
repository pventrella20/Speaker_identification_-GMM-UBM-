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
    file_names = ['AA_alberto_angela', 'AC_andrea_camilleri', 'GC_giuseppe_conte', 'GM_gianni_morandi', 'JB_papa_francesco',
                  'MB_mike_bongiorno', 'MG_mariastella_gelmini', 'MR_matteo_renzi', 'SB_silvio_berlusconi', 'SF_sabrina_ferilli',
                  'Z1_rosario_fiorello', 'Z2_maurizio_crozza', 'Z3_paola_cortellesi', 'Z4_neri_marcore', 'Z5_virginia_raffaele']
    file_names_imitations = ['01_neri_marcore_angela', '02_rosario_fiorello_bongiorno', '03_rosario_fiorello_morandi', '04_maurizio_crozza_papa',
                             'neri_marcore_angela', 'neri_marcore_angela_2', 'neri_marcore_conte', 'paola_cortellesi_gelmini',
                             'rosario_fiorello_berlusconi', 'rosario_fiorello_bongiorno', 'rosario_fiorello_morandi',
                             'rosario_fiorello_morandi2', 'virginia_raffaele_ferilli', 'maurizio_crozza_renzi', 'maurizio_crozza_berlusconi',
                             'paola_cortellesi_gelmini2', 'rosario_fiorello_camilleri', 'rosario_fiorello_camilleri2']
    file_names_imitations_2 = ['neri_marcore_angela3', 'maurizio_crozza_papa', 'neri_marcore_conte', 'paola_cortellesi_gelmini2',
                               'rosario_fiorello_camilleri', 'virginia_raffaele_ferilli', 'maurizio_crozza_berlusconi', 'maurizio_crozza_renzi',
                               'rosario_fiorello_morandi_2']
    for file in file_names_imitations_2:
        split_wav = SplitWavAudio(folder_test, file + '.wav')
        split_wav.multiple_split(sec_per_split=1)
