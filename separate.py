import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import time
import os
import librosa
import numpy as np
import IPython.display as ipd

from hparams import create_hparams
from model import Model
from soundfile import write

import gc
from tqdm import tqdm

# separate files in a dir
class Separator(object):

    def __init__(self, haprams, mix_dir, model_name, model_path):
        self.hparams = hparams
        self.test_dir = mix_dir
        self.model_name = model_name
        self.model_path = model_path
        self.datatset_name = os.path.split(mix_dir)[1]
        self.audio_list = []
        for fpath, dirs, fs in os.walk(mix_dir):
            for f in fs:
                file_name = os.path.join(fpath, f)
                self.audio_list.append(file_name)

        self.model = Model(hparams, model_name)
        self.model.load_model(model_path)

    def sep(self, ):
        for fname in tqdm(self.audio_list):
            self.separate(fname)

    def separate(self, fname):

        gc.collect()
        save_path = fname.replace(f'{self.datatset_name}', f'{self.datatset_name}_nodrums').replace('.mp3', '.wav')
        if os.path.exists(save_path):
            # print('Separated, skip...')
            return
        if os.path.split(save_path)[1].split('.')[-1] not in ['mp3', 'wav']:
            # print('Not audio file, skip...')
            return
        mix, _ = librosa.load(fname, sr=self.hparams.SR)
        insts_wav = self.model.separate(mix, self.hparams, self.model_path)
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)

        write(os.path.join(save_path), insts_wav[0], self.hparams.SR)

mix_dir = '/home/jlqian/tempo_dataset/'
mix_dir_list = [os.path.join(mix_dir, dir) for dir in os.listdir(mix_dir)]

model_name = 'mmdensenet'
model_path = '../models/mmdensenet/best.pkl'

hparams = create_hparams()
hparams.hard_mask = True
hparams.extract = False


for dir in mix_dir_list:
    if os.path.split(dir)[1][-7:] == 'nodrums':
        continue
    print(f'Separating dataset: {os.path.split(dir)[1]}...')
    separator = Separator(hparams, dir, model_name, model_path)
    separator.sep()