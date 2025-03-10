import io
import os
import numpy as np
from pymongo import MongoClient
import gridfs
from minio import Minio
from minio.error import S3Error

# Paramètres globaux
THRESHOLD = 2.2
BATCH_SIZE = 50  # Ajustez selon la RAM disponible

#####################################
# Fonctions de traitement spectrogramme
#####################################

def spectral_gate(S, noise_mean, threshold=THRESHOLD):
    """
    Applique un gating spectral : conserve les valeurs supérieures à noise_mean * threshold.
    """
    mask = S > (noise_mean[:, np.newaxis] * threshold)
    S_denoised = S * mask
    return S_denoised

def apply_distortion(S, factor=1.1):
    """
    Applique une distorsion simple en élevant le spectrogramme à la puissance 'factor'.
    """
    S_distorted = np.power(S, factor)
    return S_distorted

def process_spectrogram(S, threshold=THRESHOLD, apply_distortion_flag=True):
    """
    À partir du spectrogramme de base S, estime un profil de bruit (10ème percentile),
    applique le spectral gating et éventuellement une distorsion.
    Renvoie un tuple (S_denoised, S_distorted).
    """
    noise_mean = np.percentile(S, 10, axis=1)
    S_denoised = spectral_gate(S, noise_mean, threshold)
    if apply_distortion_flag:
        S_distorted = apply_distortion(S_denoised)
    else:
        S_distorted = S_denoised
    return S_denoised, S_distorted

def create_minio_client(endpoint, access_key, secret_key, secure=False):
    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )

##############################################
# Pipeline de curatation avec GridFS
##############################################

def curate_metadata_with_spectrograms_batch():
    # Connexion à la base source de metadata
    source_client = MongoClient("mongodb://mongodb:27017")
    source_db = source_client["birdclef"]   # Adaptez ce nom
    metadata_coll = source_db["metadata"]

    # Connexion à la base "curated" dans laquelle on va insérer les documents enrichis
    curated_client = MongoClient("mongodb://mongodb:27017")
    curated_db = curated_client["curated"]
    curated_coll = curated_db["curated_data"]
    # On crée un objet GridFS dans la collection "spectrograms" de la base "curated"
    fs = gridfs.GridFS(curated_db, collection="spectrograms")

    # Connexion au bucket staging de Minio
    staging_endpoint = "minio:9000"   # Port S3 de Minio
    staging_access_key = "minioadmin"
    staging_secret_key = "minioadmin"
    staging_bucket = "staging-bucket"
    minio_client = create_minio_client(staging_endpoint, staging_access_key, staging_secret_key, secure=False)

    # Itérer sur les documents sans charger toute la collection en mémoire
    cursor = metadata_coll.find({})
    batch_docs = []
    count_total = 0
    count_processed = 0

    for doc in cursor:
        count_total += 1
        # Supprime les champs indésirables et l'_id
        curated_doc = {k: v for k, v in doc.items() if k not in ["license", "rating", "author", "_id"]}
        
        filename = doc.get("filename")
        if not filename:
            print("Document sans filename, passage...")
            continue
        
        # Exemple : "asbfly/XC134896.ogg" → base_name = "XC134896"
        base_name = os.path.splitext(os.path.basename(filename))[0]
        npy_filename = f"{base_name}.npy"

        # Télécharger le fichier .npy depuis le bucket staging
        try:
            response = minio_client.get_object(staging_bucket, npy_filename)
            npy_data = response.read()
            response.close()
            response.release_conn()
        except S3Error as e:
            print(f"Erreur Minio pour {npy_filename} : {e}")
            continue
        except Exception as e:
            print(f"Erreur inattendue pour {npy_filename} : {e}")
            continue

        try:
            npy_buffer = io.BytesIO(npy_data)
            spec_base = np.load(npy_buffer)
        except Exception as e:
            print(f"Erreur lors du chargement de {npy_filename}: {e}")
            continue

        # Appliquer le traitement sur le spectrogramme de base
        try:
            spec_denoised, spec_distorted = process_spectrogram(spec_base, threshold=THRESHOLD, apply_distortion_flag=True)
        except Exception as e:
            print(f"Erreur lors du traitement du spectrogramme {npy_filename}: {e}")
            continue

        # Stocker chaque spectrogramme dans GridFS
        try:
            # Stocker la version base
            base_buffer = io.BytesIO()
            np.save(base_buffer, spec_base, allow_pickle=False)
            base_buffer.seek(0)
            base_file_id = fs.put(base_buffer.getvalue(),
                                  filename=f"{base_name}_base.npy",
                                  contentType="application/octet-stream")
            
            # Stocker la version denoised
            denoised_buffer = io.BytesIO()
            np.save(denoised_buffer, spec_denoised, allow_pickle=False)
            denoised_buffer.seek(0)
            denoised_file_id = fs.put(denoised_buffer.getvalue(),
                                      filename=f"{base_name}_denoised.npy",
                                      contentType="application/octet-stream")
            
            # Stocker la version distordue
            distorted_buffer = io.BytesIO()
            np.save(distorted_buffer, spec_distorted, allow_pickle=False)
            distorted_buffer.seek(0)
            distorted_file_id = fs.put(distorted_buffer.getvalue(),
                                       filename=f"{base_name}_distorted.npy",
                                       contentType="application/octet-stream")
        except Exception as e:
            print(f"Erreur lors du stockage GridFS pour {npy_filename}: {e}")
            continue

        # Ajouter dans le document les références GridFS (les IDs sont de type ObjectId)
        curated_doc["spectrogram_base_id"] = base_file_id
        curated_doc["spectrogram_denoised_id"] = denoised_file_id
        curated_doc["spectrogram_distorted_id"] = distorted_file_id
        
        batch_docs.append(curated_doc)
        count_processed += 1
        print(f"Document {base_name} traité ({count_processed} documents traités).")
        
        # Insertion par batch pour limiter l'utilisation de la RAM
        if len(batch_docs) >= BATCH_SIZE:
            curated_coll.insert_many(batch_docs)
            print(f"{len(batch_docs)} documents insérés dans 'curated.curated_data'.")
            batch_docs = []  # Réinitialisation pour le prochain batch

    # Insertion du dernier batch, s'il existe
    if batch_docs:
        curated_coll.insert_many(batch_docs)
        print(f"{len(batch_docs)} documents insérés dans 'curated.curated_data' (dernier batch).")

    print(f"Traitement terminé. Documents traités : {count_processed} sur {count_total}.")

def main():
    curate_metadata_with_spectrograms_batch()

if __name__ == "__main__":
    main()
