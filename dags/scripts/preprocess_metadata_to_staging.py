import io
import pandas as pd
from minio import Minio
from minio.error import S3Error
from pymongo import MongoClient

def ingest_csv_from_minio_to_mongodb(
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    object_name: str,
    mongo_uri: str = "mongodb://mongodb:27017",
    db_name: str = "birdclef",
    collection_name: str = "metadata",
    secure: bool = False
):
    """
    1) Se connecte à un bucket Minio (API S3) sur 'endpoint'
    2) Récupère l'objet CSV nommé 'object_name' depuis 'bucket_name'
    3) Lit le CSV en mémoire grâce à pandas
    4) Insère les lignes dans MongoDB (dans la base 'db_name', collection 'collection_name')

    Paramètres :
    - endpoint : e.g. "localhost:9000"
    - secure : False si vous n'utilisez pas HTTPS
    - object_name : e.g. "filtered_metadata.csv"
    - mongo_uri : e.g. "mongodb://localhost:27017"
    """

    print(f"Connexion à Minio sur {endpoint} (secure={secure})...")
    # 1) Création du client Minio
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )

    try:
        # On ne veut pas garder le csv en local donc on parse l'objet csv en stream et on le lit après conversion utf8
        # 2) Récupération de l'objet CSV comme un flux (stream)
        print(f"Téléchargement en mémoire de l'objet '{object_name}' dans le bucket '{bucket_name}'...")
        response = client.get_object(bucket_name, object_name)
        csv_data = response.read()  # Données brutes (bytes)
        response.close()
        response.release_conn()
    except S3Error as e:
        print(f"Erreur S3/Minio lors de la récupération du fichier : {e}")
        return
    except Exception as e:
        print(f"Erreur inattendue lors de la récupération du fichier : {e}")
        return

    try:
        # 3) Décodage en UTF-8, puis lecture via pandas
        csv_str = csv_data.decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_str))
        print(f"CSV chargé : {len(df)} lignes trouvées.")
    except Exception as e:
        print(f"Erreur lors de la lecture du CSV en pandas : {e}")
        return

    # 4) Insertion dans MongoDB
    try:
        print(f"Connexion à MongoDB : {mongo_uri}")
        client_mongo = MongoClient(mongo_uri)
        db = client_mongo[db_name]
        collection = db[collection_name]

        if df.empty:
            print("DataFrame vide : aucun document à insérer.")
            return

        # Convertir en liste de dictionnaires
        records = df.to_dict(orient='records')
        print(f"Insertion de {len(records)} documents dans '{db_name}.{collection_name}'...")

        result = collection.insert_many(records)
        print(f"{len(result.inserted_ids)} documents insérés avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'insertion dans MongoDB : {e}")
        return

def main():
    # Paramètres Minio
    endpoint = "localhost:9000" 
    access_key = "minioadmin"
    secret_key = "minioadmin"
    bucket_name = "raw-bucket"
    object_name = "filtered_metadata.csv"

    # Paramètres MongoDB
    mongo_uri = "mongodb://mongodb:27017"
    db_name = "birdclef"
    collection_name = "metadata"

    # Appel à la fonction
    ingest_csv_from_minio_to_mongodb(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=bucket_name,
        object_name=object_name,
        mongo_uri=mongo_uri,
        db_name=db_name,
        collection_name=collection_name,
        secure=False 
    )

if __name__ == "__main__":
    main()
