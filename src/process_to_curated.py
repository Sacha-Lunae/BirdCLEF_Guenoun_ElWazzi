import io
import os
import numpy as np
from minio import Minio
from minio.error import S3Error

##############################################
# Configuration et fonctions de traitement   #
##############################################

# Seuil de spectral gating
THRESHOLD = 2.2

def spectral_gate(S, noise_mean, threshold=THRESHOLD):
    """
    Applique une opération de gating spectral.
    Pour chaque fréquence (axe 0), on compare S à (noise_mean * threshold).
    On garde les valeurs supérieures, les autres sont mises à 0.
    """
    # noise_mean doit être de forme (n_freq,) et S de forme (n_freq, n_time)
    mask = S > (noise_mean[:, np.newaxis] * threshold)
    S_denoised = S * mask
    return S_denoised

def apply_distortion(S, factor=1.1):
    """
    Applique une distorsion simple en amplifiant non linéairement le signal.
    Par exemple, en élevant les valeurs à une puissance > 1.
    """
    # On suppose que S est normalisé (par exemple entre 0 et 1 ou en échelle dB normalisée).
    # Ici on lève à la puissance factor pour introduire une légère non-linéarité.
    S_distorted = np.power(S, factor)
    return S_distorted

def process_spectrogram(S, threshold=THRESHOLD, apply_distortion_flag=True):
    """
    Prend en entrée un spectrogramme S (matrice numpy),
    calcule une estimation du bruit par le 10ème percentile de chaque fréquence,
    applique la fonction de spectral gating,
    et, si demandé, applique une distorsion.
    Retourne le spectrogramme traité.
    """
    # Estimer le "bruit" par le 10ème percentile le long du temps pour chaque fréquence
    noise_mean = np.percentile(S, 10, axis=1)
    S_denoised = spectral_gate(S, noise_mean, threshold=threshold)
    
    if apply_distortion_flag:
        S_processed = apply_distortion(S_denoised)
    else:
        S_processed = S_denoised

    return S_processed

def create_minio_client(endpoint, access_key, secret_key, secure=False):
    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )

#################################################
# Fonction principale de traitement et upload   #
#################################################

def process_spectrograms_in_staging(
    staging_endpoint,
    staging_access_key,
    staging_secret_key,
    staging_bucket,
    curated_endpoint,
    curated_access_key,
    curated_secret_key,
    curated_bucket,
    secure=False
):
    """
    Parcourt tous les fichiers .npy du bucket staging.
    Pour chacun :
      - Télécharge le fichier en mémoire,
      - Charge le tableau numpy,
      - Applique la fonction de denoising (et distorsion si voulu),
      - Sérialise le résultat en .npy dans un buffer,
      - Téléverse le fichier dans le bucket curated (à la racine).
    """
    print(f"[Staging] Connexion à Minio sur {staging_endpoint}")
    staging_client = create_minio_client(staging_endpoint, staging_access_key, staging_secret_key, secure=secure)
    
    print(f"[Curated] Connexion à Minio sur {curated_endpoint}")
    curated_client = create_minio_client(curated_endpoint, curated_access_key, curated_secret_key, secure=secure)
    
    # Vérifier ou créer le bucket curated
    if not curated_client.bucket_exists(curated_bucket):
        print(f"Le bucket '{curated_bucket}' n'existe pas, création...")
        curated_client.make_bucket(curated_bucket)
    
    print(f"Parcours des objets dans le bucket '{staging_bucket}'...")
    objects = staging_client.list_objects(staging_bucket, recursive=True)
    
    for obj in objects:
        # On ne traite que les fichiers .npy
        if not obj.object_name.lower().endswith('.npy'):
            continue

        staging_object_name = obj.object_name
        print(f"Traitement de l'objet : {staging_object_name}")
        
        # Téléchargement en mémoire du fichier .npy
        try:
            response = staging_client.get_object(staging_bucket, staging_object_name)
            npy_data = response.read()
            response.close()
            response.release_conn()
        except S3Error as e:
            print(f"Erreur Minio lors de la récupération de '{staging_object_name}': {e}")
            continue
        except Exception as e:
            print(f"Erreur inattendue lors de la récupération de '{staging_object_name}': {e}")
            continue

        # Charger le tableau numpy à partir d'un buffer
        try:
            npy_buffer = io.BytesIO(npy_data)
            S = np.load(npy_buffer)
        except Exception as e:
            print(f"Erreur lors du chargement du fichier npy '{staging_object_name}': {e}")
            continue

        # Appliquer le denoising (et distorsion)
        try:
            S_processed = process_spectrogram(S, threshold=THRESHOLD, apply_distortion_flag=True)
        except Exception as e:
            print(f"Erreur lors du traitement du spectrogramme '{staging_object_name}': {e}")
            continue

        # Sérialiser le spectrogramme traité en .npy dans un buffer
        try:
            output_buffer = io.BytesIO()
            np.save(output_buffer, S_processed, allow_pickle=False)
            output_buffer.seek(0)
        except Exception as e:
            print(f"Erreur lors de la sérialisation de '{staging_object_name}': {e}")
            continue

        # Déterminer le nom de l'objet dans le bucket curated (même base, extension .npy)
        base_name = os.path.splitext(os.path.basename(staging_object_name))[0]
        curated_object_name = f"{base_name}.npy"

        # Téléversement dans le bucket curated
        try:
            curated_client.put_object(
                bucket_name=curated_bucket,
                object_name=curated_object_name,
                data=output_buffer,
                length=len(output_buffer.getvalue()),
                content_type="application/octet-stream"
            )
            print(f"Objet traité envoyé dans '{curated_bucket}/{curated_object_name}'")
        except Exception as e:
            print(f"Erreur lors du téléversement de '{curated_object_name}': {e}")
            continue

    print("Traitement de tous les spectrogrammes terminé.")

#####################
# Fonction main     #
#####################

def main():
    # Paramètres Minio pour le bucket staging
    staging_endpoint = "localhost:9000"
    staging_access_key = "minioadmin"
    staging_secret_key = "minioadmin"
    staging_bucket = "bucket-staging"

    # Paramètres Minio pour le bucket curated (même endpoint ici, mais buckets différents)
    curated_endpoint = "localhost:9000"
    curated_access_key = "minioadmin"
    curated_secret_key = "minioadmin"
    curated_bucket = "curated-bucket"

    process_spectrograms_in_staging(
        staging_endpoint=staging_endpoint,
        staging_access_key=staging_access_key,
        staging_secret_key=staging_secret_key,
        staging_bucket=staging_bucket,
        curated_endpoint=curated_endpoint,
        curated_access_key=curated_access_key,
        curated_secret_key=curated_secret_key,
        curated_bucket=curated_bucket,
        secure=False
    )

if __name__ == "__main__":
    main()
