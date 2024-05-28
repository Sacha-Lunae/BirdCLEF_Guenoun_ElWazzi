# %%
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import multiprocessing as mp
from tqdm.notebook import tqdm

# %%
df_meta = pd.read_csv("./birdclef-2024/train_metadata.csv")

# %%
df_meta

# %%
df_train = df_meta[["primary_label", "filename"]]

# %%
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

    

# %%
def extract_audio_features_with_path(args):
    ogg_file_path, max_length = args
    return extract_audio_features(ogg_file_path, max_length)

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


# %%

# Example usage
audio_features = extract_audio_features('./birdclef-2024/train_audio/asbfly/XC49755.ogg')

# Convert features to a dictionary of DataFrames for better visualization
features_df = {key: pd.DataFrame(value) for key, value in audio_features.items()}

# Display the extracted features
for feature_name, df in features_df.items():
    print(f"\nFeature: {feature_name}")
    display(df.head())  # Using display() from IPython.display for better visualization in Jupyter

# %%
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

def format_features(aggregated_features):
    formatted_features = []
    for key in aggregated_features:
        for stat in aggregated_features[key]:
            formatted_features.extend(aggregated_features[key][stat])
    return np.array(formatted_features)

# %%
ogg_file_path = './birdclef-2024/train_audio/asbfly/XC49755.ogg'
max_length = 22050 * 5  # For example, 5 seconds at a sample rate of 22050 Hz

# Extract and aggregate features
features = extract_audio_features(ogg_file_path, max_length)
aggregated_features = aggregate_features(features)

# Format features for model input
formatted_features = format_features(aggregated_features)

print(formatted_features.shape)
print(formatted_features)


# %%

def load_data(file_paths, labels, max_length=22050*5):
    X = []
    y = []

    # Prepare arguments for multiprocessing
    args = [(file_path, max_length) for file_path in file_paths]

    # Use multiprocessing to speed up the feature extraction process
    with mp.Pool(mp.cpu_count()) as pool:
        features_list = list(tqdm(pool.imap(extract_audio_features_with_path, args), total=len(args)))

    for features in features_list:
        aggregated_features = aggregate_features(features)
        formatted_features = format_features(aggregated_features)
        X.append(formatted_features)
    
    return np.array(X), np.array(labels)


# %%

# Assuming df_train is already loaded in your notebook
file_paths = df_train['filename'].apply(lambda x : "./birdclef-2024/train_audio/" + x ).tolist()
labels = df_train['primary_label'].astype('category').cat.codes.tolist()

# Load data
X, y = load_data(file_paths, labels)

# %%


# Example file paths and labels
file_paths = ['./birdclef-2024/train_audio/asbfly/XC49755.ogg', './birdclef-2024/train_audio/asbfly/XC134896.ogg']
labels = [0, 1]  # Example labels

# Load data
X, y = load_data(file_paths, labels)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert labels to categorical
y = to_categorical(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
def create_model(input_shape):
    model = Sequential([
        Dense(256, input_shape=(input_shape,), activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
model = create_model(X_train.shape[1])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")


