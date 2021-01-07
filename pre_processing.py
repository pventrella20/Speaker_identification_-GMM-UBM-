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
            split_fn = str(int(i/sec_per_split)) + '_' + self.filename
            self.single_split(i, i + sec_per_split, split_fn)
            print(str(int(i/sec_per_split)) + ' Done')
            if i == total_secs - sec_per_split:
                print('All splited successfully')

if __name__ == '__main__':
    folder = './data/gmm_dataset'
    split_wav = SplitWavAudio(folder, '*.wav')
    split_wav.multiple_split(sec_per_split=20)