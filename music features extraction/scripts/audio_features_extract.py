from audio_feature_utils import *

from tqdm import tqdm
import numpy as np
import os


def extract_features(y, sr, fps, file_path):
    step_size = 1.0 / fps

    hop = int(sr * step_size)

    features = {}

    # Chroma Features (12-dim)
    features['chroma'] = extract_features_chroma(y, sr, hop, 12)
    # features['chroma'] = extract_features_madmom_chroma(file_path, fps, sr)
    # features['chroma'] = extract_features_hybrid(y, sr, hop, 6)

    # Mel Spectrogram (80-dim)
    features['spectogram'] = extract_features_mel(y, sr, hop, 80)
    # features['spectogram'] = extract_features_madmom_mel(file_path, fps, 80,  sr)
    # features['spectogram'] = extract_features_multi_mel(y, sr, hop, mel_dim=80)

    # Spectral Flux (1-dim)
    features['flux'] = extract_features_spectral_flux(file_path, fps)

    # Beats (2-dim)
    features['beats'] = extract_features_madmom_beat(file_path, fps)

    # MFCCs (20-dim)
    features['mfcc'] = extract_features_mfcc(y, sr, hop)
    # features['mfcc'] = extract_features_madmom_mfcc(file_path, fps, 20,  sr)

    # Zero Crossing Rate (1-dim)
    features['zerocrossing'] = extract_features_zero_crossing(y, hop)

    standard_row = features['flux'].shape[1]

    adjust_rows(features['chroma'], standard_row)
    adjust_rows(features['spectogram'], standard_row)
    adjust_rows(features['beats'], standard_row)
    adjust_rows(features['mfcc'], standard_row)
    adjust_rows(features['zerocrossing'], standard_row)

    return features


def adjust_rows(arr, N):
    rows, cols = arr.shape
    if rows > N:
        arr = arr[:N, :]
    elif rows < N:
        padding = np.zeros((N - rows, cols))
        arr = np.vstack((arr, padding))
    return arr


def save_features(feature_data, file_path):
    np.savetxt(file_path, feature_data, fmt='%.7f')


def process_npz_files(input_wav_folder, sample_rate):
    output_folder = os.path.join(input_wav_folder, 'features')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    wav_files = [f for f in os.listdir(input_wav_folder) if f.endswith('.wav')]

    for i, file_name in enumerate(tqdm(wav_files, desc="Processing files", unit="file")):
        file_path = os.path.normpath(os.path.join(input_wav_folder, file_name))
        y, sr = librosa.load(file_path, sr=sample_rate)

        features = extract_features(y, sr, 60, file_path)

        # Create output directory for each audio file
        audio_name = os.path.splitext(file_name)[0]
        audio_output_folder = os.path.join(output_folder, audio_name)
        if not os.path.exists(audio_output_folder):
            os.makedirs(audio_output_folder)

        # Save each feature
        for feature_name, feature_data in features.items():
            feature_file_path = os.path.join(audio_output_folder, f'{feature_name}.txt')
            save_features(feature_data, feature_file_path)


if __name__ == '__main__':

    # test
    input_folder = '../data/wav'

    # sample rate must be divisible by 60
    sample_rate = 23040

    process_npz_files(input_folder, sample_rate)
