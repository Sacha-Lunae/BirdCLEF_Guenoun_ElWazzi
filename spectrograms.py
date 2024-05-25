import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
import IPython.display as ipd

cmap = matplotlib.colormaps['viridis']

class CFG:
    seed = 42
    
    # Input image size and batch size
    img_size = [775, 308]
    batch_size = 64
    
    # Audio duration, sample rate, and length
    duration = 15 # second
    sample_rate = 32000
    audio_len = duration * sample_rate
    
    # STFT parameters
    nfft = 2028
    window = 2048
    hop_length = audio_len // (img_size[1] - 1)
    fmin = 20
    fmax = 16000
    
    # Number of epochs, model name
    epochs = 3 # à changer quand on sera sûrs de nous
    preset = 'efficientnetv2_b2_imagenet'
    
    # Data augmentation parameters
    augment = True

    # Class Labels for BirdCLEF 24
    class_names = sorted(os.listdir('birdclef-2024/train_audio/'))
    num_classes = len(class_names)
    class_labels = list(range(num_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v: k for k, v in label2name.items()}

def load_audio(filepath):
    audio, sr = librosa.load(filepath, sr=CFG.sample_rate)
    return audio, sr


# POURQUOI LE FAIRE COMME CA ?
# + de précision, permet de mieux distinguer les différents bruits
def get_spectrogram_rgb(audio):
    spec = librosa.feature.melspectrogram(y=audio, sr=CFG.sample_rate, n_mels=256, n_fft=2048, hop_length=512, fmax=CFG.fmax, fmin=CFG.fmin)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    # Convert spectrogram to color image
    spec_img = plt.cm.viridis(spec_db)
    spec_img = (spec_img[:, :, :3] * 255).astype(np.uint8)  # Discard alpha channel and convert to 8-bit integer
    return spec_img

# POURQUOI LE FAIRE COMME CA ?
# + rapide pour faire des tests au début, à voir si on perd vraiment énormément d'info
def get_spectrogram_bw(audio):

    spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=CFG.sample_rate, 
        n_mels=256, 
        n_fft=2048, 
        hop_length=512, 
        fmax=CFG.fmax, 
        fmin=CFG.fmin
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    
    # Normalize the spectrogram
    spec_db -= spec_db.min()
    spec_db /= spec_db.max()

    # Convert to 8-bit integer
    spec_img_bw = (spec_db * 255).astype(np.uint8)
    
    return spec_img_bw

def preprocess_audio(filepath, rgb):
    audio, sr = load_audio(filepath)
    audio = audio[:CFG.audio_len]
    if rgb :
        return get_spectrogram_rgb(audio)
    else : 
        return get_spectrogram_bw(audio)

def data_generator(df, batch_size=32, rgb=False):
    while True:
        batch_data = df.sample(n=batch_size)
        spectrograms = []
        metas = []
        targets = []
        
        for _, row in batch_data.iterrows():
            spectrogram = preprocess_audio(row['filepath'], rgb)
            meta = [row['latitude'], row['longitude']]
            target = row['target']
            
            spectrograms.append(spectrogram)
            metas.append(meta)
            targets.append(target)
        
        yield [np.array(spectrograms), np.array(metas)], np.array(targets)
