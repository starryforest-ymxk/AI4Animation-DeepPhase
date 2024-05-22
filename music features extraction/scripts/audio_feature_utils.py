import librosa
import madmom as madmom
from madmom.audio.signal import Signal
from madmom.audio.filters import MelFilterbank
from madmom.audio.spectrogram import *
from madmom.audio.chroma import CLPChroma
from scipy.fftpack import dct

def extract_features_spectral_flux(music_file, tgt_fps=20):
    filtbank = madmom.audio.filters.MelFilterbank
    spec = Spectrogram(music_file, fps=tgt_fps, filterbank=filtbank, num_channels = 1)
    spectralflux = madmom.features.onsets.spectral_flux(spec)
    return np.expand_dims(spectralflux, 1)

def extract_features_madmom_beat(music_file, tgt_fps=20):
    proc_dwn = madmom.features.RNNDownBeatProcessor()
    beats = proc_dwn(music_file, fps=tgt_fps)
    return beats

def extract_features_madmom_mel(music_file, tgt_fps=20, dim=80, sample_rate = 44100):

    signal = Signal(music_file, sample_rate=sample_rate, num_channels=1)
    spectrogram = Spectrogram(signal, fps=tgt_fps)

    bin_frequencies = np.linspace(0, sample_rate / 2, spectrogram.shape[1])

    mel_filter = MelFilterbank(num_bands=dim,sample_rate=sample_rate, bin_frequencies=bin_frequencies, fmin=20, fmax=sample_rate/2 - 20, unique_filters=True)
    mel_spectrogram = LogarithmicFilteredSpectrogram(spectrogram, filterbank=mel_filter, norm_filters=True, log=np.log)

    return mel_spectrogram

def extract_features_madmom_mfcc(music_file, tgt_fps=20, dim=20, sample_rate = 44100):

    signal = Signal(music_file, sample_rate=sample_rate, num_channels=1)
    spectrogram = Spectrogram(signal, fps=tgt_fps)

    bin_frequencies = np.linspace(0, sample_rate / 2, spectrogram.shape[1])

    mel_filter = MelFilterbank(sample_rate=sample_rate, bin_frequencies=bin_frequencies, fmin=20, fmax=sample_rate/2 - 20, unique_filters=True)
    mel_spectrogram = LogarithmicFilteredSpectrogram(spectrogram, filterbank=mel_filter, norm_filters=True, log=np.log)
    db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    mfcc = dct(db, type=2, axis=1, norm='ortho')[:, : (dim + 1)]
    return mfcc[:, 1:]

def extract_features_madmom_chroma(music_file, tgt_fps=20, sample_rate = 44100):

    signal = Signal(music_file, sample_rate=sample_rate, num_channels=1)
    chroma = CLPChroma(signal)

    return chroma



def extract_features_multi_mel(y, sr=44100.0, hop=512, nffts=[1024, 2048, 4096], mel_dim=100):
    featuress = []
    for nfft in nffts:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_dim, n_fft=nfft, hop_length=hop)  # C2 is 65.4 Hz
        features = librosa.power_to_db(mel, ref=np.max)
        featuress.append(features)
    features = np.stack(featuress, axis=1)
    return features

def extract_features_hybrid(y,sr,hop,mel_dim=12,window_mult=1):
    hop -= hop % 32  #  Chroma CQT only accepts hop lengths that are multiples of 32, so this ensures that condition is met
    window = window_mult * hop # Fast Fourier Transform Window Size is a multiple (default 1) of the hop
    y_harm, y_perc = librosa.effects.hpss(y)
    mels = librosa.feature.melspectrogram(y=y_perc, sr=sr,n_fft=window,hop_length=hop,n_mels=mel_dim)
    cqts = librosa.feature.chroma_cqt(y=y_harm, sr=sr,hop_length= hop,
                                      norm=np.inf, threshold=0, n_chroma=mel_dim,
                                      n_octaves=6, fmin=65.4, cqt_mode='full')
    joint = np.concatenate((mels, cqts), axis=0)
    return joint.T


def extract_features_mel(y, sr, hop, mel_dim=100):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_dim, hop_length=hop)
    features = librosa.power_to_db(mel, ref=np.max)
    return features.T

def extract_features_envelope(y, sr, hop, mel_dim=100):
    envelope = librosa.onset.onset_strength(y=y, hop_length=hop, n_mels=mel_dim)
    return np.expand_dims(envelope,1)

def extract_features_chroma(y, sr, hop, dim):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, n_chroma=dim)
    return chroma.T

def extract_features_mfcc(y,sr,hop):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
    return mfcc.T

def extract_features_zero_crossing(y, hop):
    zero_crossings = librosa.feature.zero_crossing_rate(y=y , hop_length=hop)
    return zero_crossings.T