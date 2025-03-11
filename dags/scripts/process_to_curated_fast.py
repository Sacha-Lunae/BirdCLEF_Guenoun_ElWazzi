import io
import os
import numpy as np
from pymongo import MongoClient
import gridfs
from minio import Minio
from minio.error import S3Error
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paramètres globaux
THRESHOLD = 2.2
BATCH_SIZE = 50  # Ajustez selon la RAM disponible
MAX_WORKERS = 8  # Ajustez en fonction de la charge et de l'environnement

#####################################
# Fonctions de traitement spectrogramme
#####################################

def spectral_gate(S, noise_mean, threshold=THRESHOLD):
    """Applique un gating spectral : conserve les valeurs supérieures à noise_mean * threshold."""
    mask = S > (noise_mean[:, np.newaxis] * threshold)
    return S * mask

def apply_distortion(S, factor=1.1):
    """Applique une distorsion en élevant le spectrogramme à la puissance 'factor'."""
    return np.power(S, factor)

def process_spectrogram(S, threshold=THRESHOLD, apply_distortion_flag=True):
    """
    Calcule le profil de bruit (10ème percentile), applique le spectral gating et éventuellement une distorsion.
    Retourne un tuple (S_denoised, S_distorted).
    """
    noise_mean = np.percentile(S, 10, axis=1)
    S_denoised = spectral_gate(S, noise_mean, threshold)
    if apply_distortion_flag:
        S_distorted = apply_distortion(S_denoised)
    else:
        S_distorted = S_denoised
    return S_denoised, S_distorted

def create_minio_client(endpoint, access_key, secret_key, secure=False):
    return Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

##############################################
# Traitement d'un document et insertion dans GridFS
##############################################

def process_document(doc, minio_client, fs):
    """
    Traite un document de metadata :
      - Extrait le nom de base depuis 'filename'
      - Télécharge le fichier .npy depuis MinIO
      - Charge le spectrogramme et le traite
      - Stocke les versions dans GridFS et retourne un document enrichi.
    En cas d'erreur, renvoie None.
    """
    filename = doc.get("filename")
    if not filename:
        print("Document sans filename, passage...")
        return None

    base_name = os.path.splitext(os.path.basename(filename))[0]
    npy_filename = f"{base_name}.npy"

    # Télécharger le fichier .npy depuis le bucket staging
    try:
        response = minio_client.get_object("staging-bucket", npy_filename)
        npy_data = response.read()
        response.close()
        response.release_conn()
    except S3Error as e:
        print(f"Erreur Minio pour {npy_filename} : {e}")
        return None
    except Exception as e:
        print(f"Erreur inattendue pour {npy_filename} : {e}")
        return None

    # Charger le fichier .npy et traiter le spectrogramme
    try:
        npy_buffer = io.BytesIO(npy_data)
        spec_base = np.load(npy_buffer)
    except Exception as e:
        print(f"Erreur lors du chargement de {npy_filename}: {e}")
        return None

    try:
        spec_denoised, spec_distorted = process_spectrogram(spec_base, threshold=THRESHOLD, apply_distortion_flag=True)
    except Exception as e:
        print(f"Erreur lors du traitement du spectrogramme {npy_filename}: {e}")
        return None

    # Stocker les fichiers dans GridFS
    try:
        # Fonction utilitaire pour sauvegarder un array dans GridFS
        def save_array(array, suffix):
            buf = io.BytesIO()
            np.save(buf, array, allow_pickle=False)
            buf.seek(0)
            return fs.put(buf.getvalue(), filename=f"{base_name}_{suffix}.npy", contentType="application/octet-stream")
        
        base_file_id = save_array(spec_base, "base")
        denoised_file_id = save_array(spec_denoised, "denoised")
        distorted_file_id = save_array(spec_distorted, "distorted")
    except Exception as e:
        print(f"Erreur lors du stockage GridFS pour {npy_filename}: {e}")
        return None

    # Préparer le document enrichi (exclut certains champs)
    curated_doc = {k: v for k, v in doc.items() if k not in ["license", "rating", "author", "_id"]}
    curated_doc["spectrogram_base_id"] = base_file_id
    curated_doc["spectrogram_denoised_id"] = denoised_file_id
    curated_doc["spectrogram_distorted_id"] = distorted_file_id

    print(f"Document {base_name} traité avec succès.")
    return curated_doc

##############################################
# Pipeline de curatation avec GridFS (parallélisé)
##############################################

def curate_metadata_with_spectrograms_batch():
    # Connexion à MongoDB (source et destination)
    mongo_uri = "mongodb://mongodb:27017"
    source_client = MongoClient(mongo_uri)
    source_db = source_client["birdclef"]
    metadata_coll = source_db["metadata"]

    curated_client = MongoClient(mongo_uri)
    curated_db = curated_client["birdclef"]
    curated_coll = curated_db["curated_data"]
    fs = gridfs.GridFS(curated_db, collection="spectrograms")

    # Connexion au bucket staging de MinIO
    minio_client = create_minio_client("minio:9000", "minioadmin", "minioadmin", secure=False)

    cursor = metadata_coll.find({})
    batch_docs = []
    count_total = 0
    count_processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Soumettre le traitement de chaque document
        future_to_doc = {executor.submit(process_document, doc, minio_client, fs): doc for doc in cursor}
        for future in as_completed(future_to_doc):
            count_total += 1
            result = future.result()
            if result is not None:
                batch_docs.append(result)
                count_processed += 1
                print(f"Total traité: {count_processed} documents.")
                if len(batch_docs) >= BATCH_SIZE:
                    curated_coll.insert_many(batch_docs)
                    print(f"{len(batch_docs)} documents insérés dans 'curated.curated_data'.")
                    batch_docs = []

    # Insertion du dernier batch
    if batch_docs:
        curated_coll.insert_many(batch_docs)
        print(f"{len(batch_docs)} documents insérés dans 'curated.curated_data' (dernier batch).")

    print(f"Traitement terminé. Documents traités : {count_processed} sur {count_total}.")

def main():
    curate_metadata_with_spectrograms_batch()

if __name__ == "__main__":
    main()
