from librosa.core import stft, istft
import numpy as np


def align(length):
    tmp = 2
    while length > tmp:
        tmp = tmp * 2
    return tmp - length

def wave2spec(C, y):
    tmp = []
    if (len(y) % C.H != 0):
        pad_size = C.H - len(y) % C.H
        tmp = np.zeros(pad_size)
        y = np.append(y, tmp)
    spec = stft(y, n_fft=C.fft_size, hop_length=C.H, win_length=C.fft_size)
    mag = np.abs(spec)
    mag /= np.max(mag)
    phase = np.exp(1.j * np.angle(spec))
    return mag, phase, len(tmp)

def spec2wave(C, mag, phase, tmplen):
    y = istft(mag * phase, hop_length=C.H, win_length=C.fft_size)
    y = y[:len(y) - tmplen]
    return y / (np.max(np.abs(y))+1e-10)