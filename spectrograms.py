import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
import librosa.display
from PIL import Image

class CFG:
    seed = 42
    img_size = (256, 938)
    batch_size = 64
    duration = 15  # seconds
    sample_rate = 32000
    audio_len = duration * sample_rate
    nfft = 2028
    window = 2048
    hop_length = audio_len // (img_size[1] - 1)
    fmin = 20
    fmax = 16000
    epochs = 3
    preset = 'efficientnetv2_b2_imagenet'
    augment = True
    class_names = sorted(os.listdir('train_audio/'))
    num_classes = len(class_names)
    class_labels = list(range(num_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v: k for k, v in label2name.items()}

def load_audio(filepath):
    audio, sr = librosa.load(filepath, sr=CFG.sample_rate)
    return audio, sr

def get_spectrogram_rgb(audio):
    try:
        spec = librosa.feature.melspectrogram(y=audio, sr=CFG.sample_rate, n_mels=256, n_fft=2048, hop_length=CFG.hop_length, fmax=CFG.fmax, fmin=CFG.fmin)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_img = plt.cm.viridis(spec_db)
        spec_img = (spec_img[:, :, :3] * 255).astype(np.uint8)
        return spec_img
    except Exception as e:
        print(f"Error in get_spectrogram_rgb: {e}")
        return None

def get_spectrogram_bw(audio):
    try:
        spec = librosa.feature.melspectrogram(y=audio, sr=CFG.sample_rate, n_mels=256, n_fft=2048, hop_length=CFG.hop_length, fmax=CFG.fmax, fmin=CFG.fmin)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_db -= spec_db.min()
        spec_db /= spec_db.max()
        spec_img_bw = (spec_db * 255).astype(np.uint8)
        return spec_img_bw
    except Exception as e:
        print(f"Error in get_spectrogram_bw: {e}")
        return None

def preprocess_audio(filepath, rgb):
    try:
        audio, sr = load_audio(filepath)
        audio = audio[:CFG.audio_len]
        if rgb:
            return get_spectrogram_rgb(audio)
        else:
            return get_spectrogram_bw(audio)
    except Exception as e:
        print(f"Error in preprocess_audio: {e}")
        return None

def data_generator(df, batch_size, img_size=(256, 938), rgb=False):
    while True:
        batch_paths = np.random.choice(a=df.filepath.values, size=batch_size)
        batch_input_img = []
        batch_output = []

        for input_path in batch_paths:
            try:
                spectrogram = preprocess_audio(input_path, rgb)
                if spectrogram is not None:
                    # Ensure spectrogram has 3 dimensions (height, width, channels)
                    if spectrogram.ndim == 2:
                        spectrogram = np.expand_dims(spectrogram, axis=-1)

                    # Resize using PIL
                    if spectrogram.shape[-1] == 1:  # Grayscale
                        spectrogram_image = Image.fromarray(spectrogram[:, :, 0], mode='L')
                    else:  # RGB
                        spectrogram_image = Image.fromarray(spectrogram, mode='RGB')

                    spectrogram_image = spectrogram_image.resize((img_size[1], img_size[0]), Image.LANCZOS)
                    spectrogram = np.array(spectrogram_image)

                    # Ensure final shape matches expected shape
                    if spectrogram.ndim == 2:
                        spectrogram = np.expand_dims(spectrogram, axis=-1)  # Grayscale
                    elif spectrogram.ndim == 3 and spectrogram.shape[-1] == 3:
                        pass  # RGB

                    batch_input_img.append(spectrogram)
                    target = df[df.filepath == input_path]['target'].values[0]
                    batch_output.append(target)
                else:
                    print(f"Failed to process file: {input_path}")
            except Exception as e:
                print(f"Error in data_generator for {input_path}: {e}")

        if len(batch_input_img) > 0:
            batch_input_img = np.array(batch_input_img)
            batch_output = np.array(batch_output)
            yield batch_input_img, batch_output
        else:
            print("No valid spectrograms in this batch.")