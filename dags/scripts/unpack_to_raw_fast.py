import os
import zipfile
import requests
from minio import Minio
from minio.error import S3Error
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_file(url, local_path, chunk_size=8192, session=None):
    """
    Télécharge un fichier depuis 'url' et l'enregistre sous 'local_path'.
    """
    session = session or requests.Session()
    print(f"Téléchargement de {url} --> {local_path}")
    with session.get(url, stream=True) as response:
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    print("Téléchargement terminé.")

def download_data_locally(audio_url, metadata_url, raw_dir="raw"):
    """
    - Crée le dossier 'raw_dir'
    - Télécharge l'archive audio (ZIP) et la décompresse
    - Télécharge le fichier de métadonnées (CSV)
    """
    os.makedirs(raw_dir, exist_ok=True)
    session = requests.Session()

    # Téléchargement de l'archive audio (ZIP)
    audio_zip_path = os.path.join(raw_dir, "filtered_audios.zip")
    download_file(audio_url, audio_zip_path, session=session)

    # Décompression du ZIP dans raw/filtered_audios/
    extract_dir = os.path.join(raw_dir, "filtered_audios")
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Décompression de {audio_zip_path} dans {extract_dir} ...")
    with zipfile.ZipFile(audio_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Décompression terminée.")
    os.remove(audio_zip_path)

    # Téléchargement du fichier de métadonnées (CSV)
    metadata_path = os.path.join(raw_dir, "filtered_metadata.csv")
    download_file(metadata_url, metadata_path, session=session)

def upload_file(client, bucket_name, local_path, object_name):
    """
    Téléverse un fichier unique dans le bucket MinIO.
    """
    try:
        print(f"Téléversement de {local_path} --> bucket '{bucket_name}' (objet : '{object_name}')")
        client.fput_object(bucket_name=bucket_name, object_name=object_name, file_path=local_path)
        print(f"Téléversement réussi pour {object_name}")
    except S3Error as e:
        print(f"Erreur lors du téléversement de {object_name}: {e}")

def upload_to_minio(endpoint, access_key, secret_key, bucket_name, raw_dir="raw"):
    """
    Envoie tous les fichiers du dossier 'raw_dir' dans le bucket MinIO.
    """
    print(f"Connexion à Minio sur {endpoint}...")
    client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    if not client.bucket_exists(bucket_name):
        print(f"Le bucket '{bucket_name}' n'existe pas, création...")
        client.make_bucket(bucket_name)
    else:
        print(f"Le bucket '{bucket_name}' existe déjà.")

    # Récupère tous les fichiers dans raw_dir
    files_to_upload = []
    for root, dirs, files in os.walk(raw_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            # Chemin relatif dans le bucket
            relative_path = os.path.relpath(local_path, raw_dir).replace('\\', '/')
            files_to_upload.append((local_path, relative_path))

    print(f"{len(files_to_upload)} fichiers trouvés dans '{raw_dir}'.")
    # Paralléliser le téléversement
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(upload_file, client, bucket_name, local_path, relative_path)
            for local_path, relative_path in files_to_upload
        ]
        for future in as_completed(futures):
            future.result()  # pour propager d'éventuelles exceptions

    print("Téléversement terminé.")

def main():
    # Liens Dropbox de téléchargement direct
    audio_url = "https://www.dropbox.com/scl/fi/l71wggm0sjdh2aqx7poh8/bird_clef_audios.zip?rlkey=84rhw9zkus4irs133nhm7eacw&st=kwzbw15b&dl=1"
    metadata_url = "https://www.dropbox.com/scl/fi/xy5l60a8pl1nmucd9hkat/audios_metadata.csv?rlkey=zalswdvfae9wo7m9lnt9gavk4&st=6yw9lf20&dl=1"

    # 1) Téléchargement local et décompression
    download_data_locally(audio_url, metadata_url, raw_dir="raw-test")

    # 2) Paramètres MinIO
    endpoint = "minio:9000"  # Utiliser le nom de service Docker
    access_key = "minioadmin"
    secret_key = "minioadmin"
    bucket_name = "raw-bucket"

    # Téléversement
    upload_to_minio(endpoint, access_key, secret_key, bucket_name, raw_dir="raw-test")

if __name__ == "__main__":
    main()
