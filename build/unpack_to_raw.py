import os
import zipfile
import requests
from minio import Minio
from minio.error import S3Error

def download_file(url, local_path):
    """
    Télécharge un fichier depuis 'url' et l'enregistre sous 'local_path'.
    """
    print(f"Téléchargement de {url} --> {local_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Vérifie qu'il n'y a pas d'erreur HTTP
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Téléchargement terminé.")

def download_data_locally(audio_url, metadata_url, raw_dir="raw"):
    """
    - Crée le dossier 'raw_dir'
    - Télécharge l'archive audio (ZIP) et la décompresse
    - Télécharge le fichier de métadonnées (CSV)
    """
    os.makedirs(raw_dir, exist_ok=True)

    # 1) Téléchargement de l'archive audio (ZIP)
    audio_zip_path = os.path.join(raw_dir, "filtered_audios.zip")
    print(f"Audio zip path: {audio_zip_path}")
    download_file(audio_url, audio_zip_path)

    # 2) Décompression du ZIP dans raw/filtered_audios/
    extract_dir = os.path.join(raw_dir, "filtered_audios")
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Décompression de {audio_zip_path} dans {extract_dir} ...")
    with zipfile.ZipFile(audio_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Décompression terminée.")

    # Suppression du ZIP
    os.remove(audio_zip_path)

    # 3) Téléchargement du fichier de métadonnées (CSV)
    metadata_path = os.path.join(raw_dir, "filtered_metadata.csv")
    download_file(metadata_url, metadata_path)


def upload_to_minio(endpoint, access_key, secret_key, bucket_name, raw_dir="raw"):
    """
    Envoie tous les fichiers du dossier 'raw_dir' dans le bucket Minio.
    """
    print(f"Connexion à Minio sur {endpoint}...")
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )

    found = client.bucket_exists(bucket_name)
    if not found:
        print(f"Le bucket '{bucket_name}' n'existe pas, création...")
        client.make_bucket(bucket_name)
    else:
        print(f"Le bucket '{bucket_name}' existe déjà.")

    for root, files in os.walk(raw_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            # Construire le chemin d'objet dans le bucket
            # (Par ex. "filtered_audios/XCxxxx.ogg" ou "filtered_metadata.csv")
            relative_path = os.path.relpath(local_path, raw_dir).replace('\\', '/')

            print(f"Téléversement de {local_path} --> bucket '{bucket_name}' (objet : '{relative_path}')")
            client.fput_object(
                bucket_name=bucket_name,
                object_name=relative_path,
                file_path=local_path
            )

    print("Téléversement terminé.")

def main():
    # Vos liens Dropbox de téléchargement direct
    audio_url = "https://www.dropbox.com/scl/fi/l71wggm0sjdh2aqx7poh8/bird_clef_audios.zip?rlkey=84rhw9zkus4irs133nhm7eacw&st=kwzbw15b&dl=1"
    metadata_url = "https://www.dropbox.com/scl/fi/xy5l60a8pl1nmucd9hkat/audios_metadata.csv?rlkey=zalswdvfae9wo7m9lnt9gavk4&st=6yw9lf20&dl=1"

    # 1) Téléchargement local + décompression
    download_data_locally(audio_url, metadata_url, raw_dir="raw-test")

    # 2) Paramètres Minio
    endpoint = "localhost:9000"
    access_key = "minioadmin"
    secret_key = "minioadmin"
    bucket_name = "raw-bucket"

    # Téléversement
    upload_to_minio(endpoint, access_key, secret_key, bucket_name, raw_dir="raw-test")

if __name__ == "__main__":
    main()
