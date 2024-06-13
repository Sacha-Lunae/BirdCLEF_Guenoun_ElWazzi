import os
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load bird data
taxonomy_df = pd.read_csv('./birdclef-2024/eBird_Taxonomy_v2021.csv')
sample_df = pd.read_csv('/Users/Rami/OneDrive/Desktop/efrei M1/S8/ML2/Projet/BirdCLEF_Guenoun_ElWazzi/birdclef-2024/sample_submission.csv')

# Load model
model = tf.keras.models.load_model('/Users/Rami/OneDrive/Desktop/efrei M1/S8/ML2/Projet/BirdCLEF_Guenoun_ElWazzi/models/27_05_2024_23-48/40percent.keras')

# Ensure the database folder exists
db_folder = './audio_db'
if not os.path.exists(db_folder):
    os.makedirs(db_folder)

# Function to record audio
def record_audio():
    name = name_entry.get()
    if not name:
        messagebox.showwarning("Input Error", "Please enter a name")
        return

    fs = 44100  # Sample rate
    seconds = 5  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished

    # Repeat the recording 5 times
    myrecording = np.tile(myrecording, (5, 1))

    file_path = os.path.join(db_folder, f'{name}.wav')
    write(file_path, fs, myrecording)  # Save as WAV file
    messagebox.showinfo("Recording", f"Recording saved as {file_path}")

    # Automatically identify the bird after recording
    identify_bird(name)

# Function to extract audio features
def extract_audio_features(ogg_file_path, max_length=22050*5):
    y, sr = librosa.load(ogg_file_path, sr=None)
    
    # Ensure the audio is of fixed length
    if len(y) < max_length:
        y = np.pad(y, (0, max_length - len(y)), 'constant')
    else:
        y = y[:max_length]
    
    # Extract features
    features = {}

    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    features['mel_spectrogram'] = S_dB

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc'] = mfcc

    # Chroma feature
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma'] = chroma

    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast'] = spectral_contrast

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    features['tonnetz'] = tonnetz

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid'] = spectral_centroid

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth'] = spectral_bandwidth

    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff'] = spectral_rolloff

    # Zero crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    features['zero_crossing_rate'] = zero_crossing_rate

    # RMS
    rms = librosa.feature.rms(y=y)
    features['rms'] = rms

    return features

# Function to aggregate features
def aggregate_features(features):
    aggregated_features = {}
    for key, value in features.items():
        aggregated_features[key] = {
            'mean': np.mean(value, axis=1),
            'std': np.std(value, axis=1),
            'min': np.min(value, axis=1),
            'max': np.max(value, axis=1)
        }
    return aggregated_features

# Function to format features
def format_features(aggregated_features):
    formatted_features = []
    for key in aggregated_features:
        for stat in aggregated_features[key]:
            formatted_features.extend(aggregated_features[key][stat])
    return np.array(formatted_features).reshape(1, -1)

# Function to preprocess audio
def preprocess_audio(audio_path):
    features = extract_audio_features(audio_path)
    aggregated_features = aggregate_features(features)
    formatted_features = format_features(aggregated_features)
    return formatted_features

# Function to predict bird species from audio
def identify_bird(name):
    global bird_code
    audio_path = os.path.join(db_folder, f'{name}.wav')
    print(audio_path)
    if not os.path.exists(audio_path):
        messagebox.showerror("File Error", f"No recording found for {name}")
        return
    
    processed_audio = preprocess_audio(audio_path)
    prediction = model.predict(processed_audio)
    max_index = np.argmax(prediction[0])

    bird_code = sample_df.columns[max_index + 1]  # Get the bird code from sample_submission
    bird_info = taxonomy_df[taxonomy_df['SPECIES_CODE'] == bird_code].iloc[0]
    
    result_text = f"Your bird code is {bird_info['SPECIES_CODE']}\n"
    result_text += f"Your bird is {bird_info['PRIMARY_COM_NAME']}\n"
    result_text += f"It belongs to the family of {bird_info['FAMILY']}."
    
    result_label.config(text=result_text)

# Function to play bird sound
def play_bird_sound():
    global bird_code
    if not bird_code:
        messagebox.showwarning("No Bird Selected", "Please identify the bird first")
        return

    bird_folder = os.path.join('./birdclef-2024/train_audio', bird_code)
    if not os.path.exists(bird_folder):
        messagebox.showerror("File Error", f"No audio files found for bird code {bird_code}")
        return

    ogg_files = [f for f in os.listdir(bird_folder) if f.endswith('.ogg')]
    if not ogg_files:
        messagebox.showerror("File Error", f"No .ogg files found in folder {bird_folder}")
        return

    first_ogg_file = os.path.join(bird_folder, ogg_files[0])
    pygame.mixer.music.load(first_ogg_file)
    pygame.mixer.music.play()

# Set up GUI
root = tk.Tk()
root.title("Bird Sound Recognition")

# Centering the window
window_width = 600
window_height = 400
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)

root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

# Creating frames for better layout management
frame = tk.Frame(root)
frame.pack(expand=True)

# Adding widgets to the frame
tk.Label(frame, text="What's your name?", font=('Helvetica', 14)).pack(pady=10)
name_entry = tk.Entry(frame, font=('Helvetica', 14))
name_entry.pack(pady=10)

record_button = tk.Button(frame, text="Start recording", command=record_audio, font=('Helvetica', 14))
record_button.pack(pady=10)

result_label = tk.Label(frame, text="Your bird is", font=('Helvetica', 14))
result_label.pack(pady=10)

predict_button = tk.Button(frame, text="Identify Bird", command=lambda: identify_bird(name_entry.get()), font=('Helvetica', 14))
predict_button.pack(pady=10)

listen_button = tk.Button(frame, text="Listen to your bird", command=play_bird_sound, font=('Helvetica', 14))
listen_button.pack(pady=10)

bird_code = None

root.mainloop()
