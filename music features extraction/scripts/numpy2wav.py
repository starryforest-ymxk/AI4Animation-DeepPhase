import os
import numpy as np
import soundfile as sf


def numpy2wav(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.npz'):
            file_path = os.path.join(input_folder, filename)
            data = np.load(file_path)
            audio = data['x']
            samplerate = data['sr']

            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)

            output_path = os.path.join(output_folder, filename[:-4] + '.wav')
            # write into wav

            sf.write(output_path, audio, samplerate)

            print(f'Converted {filename} to WAV and saved to {output_path}')


if __name__ == '__main__':

    input_folder = '../data'

    output_folder = '../data/wav'

    numpy2wav(input_folder, output_folder)
