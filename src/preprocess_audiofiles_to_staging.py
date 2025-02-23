import io
import os
import numpy as np
import librosa
from PIL import Image
from minio import Minio
from minio.error import S3Error

###################################
# Paramètres et Fonctions de base #
###################################

class CFG:
    # Paramètres pour le calcul du mel-spectrogramme
    n_mels = 256
    n_fft = 2048
    hop_length = 512
    fmin = 20
    fmax = 16000

def get_spectrogram_bw(audio, sr):
    """
    Calcule un mel-spectrogramme (échelle dB) en niveaux de gris (uint8 [0..255])
    à partir d'un signal audio et de sa fréquence d'échantillonnage.
    """
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=CFG.n_mels,
        n_fft=CFG.n_fft,
        hop_length=CFG.hop_length,
        fmin=CFG.fmin,
        fmax=CFG.fmax
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)

    # Normalisation 0..1 puis passage en [0..255]
    spec_db -= spec_db.min()
    spec_db /= spec_db.max()
    spec_img_bw = (spec_db * 255).astype(np.uint8)

    return spec_img_bw

def create_minio_client(endpoint, access_key, secret_key, secure=False):
    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )

##########################################
# Fonction principale de prétraitement   #
# (PNG + NPY) sans écriture locale       #
##########################################

def preprocess_all_audios_in_bucket(
    raw_endpoint,
    raw_access_key,
    raw_secret_key,
    raw_bucket,
    staging_endpoint,
    staging_access_key,
    staging_secret_key,
    staging_bucket,
    secure=False
):
    """
    1) Parcourt tous les objets du bucket 'raw_bucket'.
    2) Pour chaque fichier audio, calcule un spectrogramme (bw) et stocke
       - un PNG (visualisation)
       - un .npy (matrice de spectrogramme)
       dans le bucket 'staging_bucket'.
    3) Ne force pas le sample rate et ne tronque pas l'audio.
    """

    # 1) Connexion aux deux clients Minio
    print(f"[Raw] Connexion à Minio: {raw_endpoint}")
    raw_client = create_minio_client(raw_endpoint, raw_access_key, raw_secret_key, secure=secure)
    print(f"[Staging] Connexion à Minio: {staging_endpoint}")
    staging_client = create_minio_client(staging_endpoint, staging_access_key, staging_secret_key, secure=secure)

    # Vérifie/crée le staging bucket si besoin
    if not staging_client.bucket_exists(staging_bucket):
        print(f"Le bucket '{staging_bucket}' n'existe pas, création...")
        staging_client.make_bucket(staging_bucket)

    # 2) Listage des objets dans le bucket 'raw_bucket'
    print(f"Parcours des objets dans le bucket '{raw_bucket}'...")
    objects = raw_client.list_objects(raw_bucket, recursive=True)

    for obj in objects:
        object_name = obj.object_name
        print(f"Traitement de l'objet : {object_name}")

        # Téléchargement en mémoire
        try:
            response = raw_client.get_object(raw_bucket, object_name)
            audio_data = response.read()
            response.close()
            response.release_conn()
        except S3Error as e:
            print(f"Erreur Minio (get_object) sur '{object_name}' : {e}")
            continue
        except Exception as e:
            print(f"Erreur inattendue (lecture) '{object_name}': {e}")
            continue

        # Charger l'audio avec librosa sans forcer le sample rate
        try:
            in_mem_file = io.BytesIO(audio_data)
            audio, sr = librosa.load(in_mem_file, sr=None)  # sr=None => conserve SR d'origine
        except Exception as e:
            print(f"Erreur lors du chargement audio (librosa) '{object_name}': {e}")
            continue

        # Calcul du spectrogramme
        try:
            spec_img_bw = get_spectrogram_bw(audio, sr)  # np.array (uint8)
        except Exception as e:
            print(f"Erreur lors du calcul du spectrogramme '{object_name}': {e}")
            continue

        # Préparer les noms (racine identique, extensions .png et .npy)
        base_name = os.path.splitext(os.path.basename(object_name))[0]
        png_name = f"{base_name}.png"
        npy_name = f"{base_name}.npy"

        ##############################
        # 1) Sauvegarde du PNG en mémoire, upload
        ##############################
        try:
            pil_img = Image.fromarray(spec_img_bw)  # image en niveaux de gris
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format="PNG")
            img_buffer.seek(0)  # Remettre le pointeur au début

            staging_client.put_object(
                staging_bucket,
                png_name,
                data=img_buffer,
                length=len(img_buffer.getvalue()),
                content_type="image/png"
            )
            print(f"Spectrogramme PNG envoyé : {staging_bucket}/{png_name}")
        except Exception as e:
            print(f"Erreur lors de la conversion/envoi PNG '{object_name}': {e}")
            continue

        ##############################
        # 2) Sauvegarde de la matrice en .npy, upload
        ##############################
        try:
            # spec_img_bw est un np.array (height, width), uint8
            # Si vous préférez garder la version float, il faut
            # avant la conversion en uint8 (cf. get_spectrogram_bw).
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, spec_img_bw, allow_pickle=False)
            npy_buffer.seek(0)

            staging_client.put_object(
                staging_bucket,
                npy_name,
                data=npy_buffer,
                length=len(npy_buffer.getvalue()),
                content_type="application/octet-stream"  # ou "application/x-npy" si vous préférez
            )
            print(f"Spectrogramme NPY envoyé : {staging_bucket}/{npy_name}")
        except Exception as e:
            print(f"Erreur lors de la conversion/envoi NPY '{object_name}': {e}")
            continue

    print("Traitement terminé.")


################
# Exemple main #
################

def main():
    # Paramètres Minio "raw"
    raw_endpoint = "localhost:9000"
    raw_access_key = "minioadmin"
    raw_secret_key = "minioadmin"
    raw_bucket = "bucket-raw"

    # Paramètres Minio "staging"
    staging_endpoint = "localhost:9000"
    staging_access_key = "minioadmin"
    staging_secret_key = "minioadmin"
    staging_bucket = "staging-bucket"

    preprocess_all_audios_in_bucket(
        raw_endpoint=raw_endpoint,
        raw_access_key=raw_access_key,
        raw_secret_key=raw_secret_key,
        raw_bucket=raw_bucket,
        staging_endpoint=staging_endpoint,
        staging_access_key=staging_access_key,
        staging_secret_key=staging_secret_key,
        staging_bucket=staging_bucket,
        secure=False
    )

if __name__ == "__main__":
    main()
