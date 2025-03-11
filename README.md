# Data Lakes Project - Rami El Wazzi & Sacha Guenoun

## Contexte

Ce projet se fait dans la continuité d'un projet de Machine Learning datant de M1, en mai 2024. Ce projet avait pour but de participer au concours kaggle BirdClef. Le concours BirdClef est un concours annuel organisé par google où l'objectif est de permettre aux chercheurs de classifier des oiseaux grâce à leurs audios et quelques éléments de contexte en metadata. 

Désormais, l'objectif est d'automatiser l'ingestion et le traitement des données dans une structure datalake. 

## Choix d'architecture

**Couche raw - bucket minio**

Nous retenons uniquement 5% de l'intégralité des fichiers proposés par le concours birdclef 2024 durant ce projet. Ce choix se fait pour des raisons de simplicité : ne pas alourdir les machines et permettre un temps de traitement raisonnable. En tout, ce sont 1240 fichiers audios qui sont intégrés, ainsi qu'un fichier csv de metadata sur ces fichiers audios. 

Ces fichiers sont récupérés via un lien one drive : nous les stockons en local (pour permettre d'explorer les données plus facilement si besoin) et dans un bucket minio *raw-bucket*.

**Couche staging - bucket minio & base mongodb**

Nous n'accordons ici pas le même traitement aux fichiers audio vs les metadata.

- Pour les fichiers audios : nous les transformons dans un premier temps en spectrogrammes via des fichiers .npy (matrices), puis nous sauvegardons une version PNG de ces audios. Dans notre projet Machine Learning de mai 2024, notre première piste était d'utiliser les PNGs via computer vision. Finalement, c'était plus simple pour nous de passer directement par les matrices. Cependant, nous pourrions décider de faire de la computer vision dans le futur, c'est pourquoi on garde les PNGs dans la couche staging. Stagining et non curated car ces imgages auraient très sûrement besoin d'un peu plus de preprocessing si l'on comptait vraiment les utiliser pour un training. Les fichiers .npy et .png sont stockés dans un bucket minio *staging-bucket*.

- Pour les metadata : nous les avons stocké dans mongo db car notre objectif à terme est d'avoir toutes les données curated sous mongo db. Les metadata sont disponibles dans la base *birdclef.metadata*.

**Couche curated - base mongodb**

Des transformations sont faites à partir des spectrogrammes .npy depuis notre bucket staging : 
- Un denoising
- Une distorsion, pour pouvoir utiliser des techniques d'augmentation lors de notre training. Initialement, nous avions prévu de faire plusieurs distorsions, mais nous n'en gardons qu'une ici pour des raisons de simplicité et pour ne pas prendre trop de temps à compute.
- Une évaluation de la qualité : si l'on estime que le spectrogramme initial ne contient pas trop de bruit, on lui attribue is_quality_audio = true. Certains notebooks gagnants du concours kaggle utilisaient une métrique simmilaire.

*Note : nous avions aussi pensé à évaluer le taux de décibels mais la compression initiale des audios rendait la tâche plus complexe que prévue.*

On retrouve ces données propres dans 3 bases mongodb : 
- *birdclef.curated_data* contient pour chaque audio : son id, ses metadata, la métrique is_quality_audio, ainsi que 3 ids correspondant aux 3 spectrogrammes .npy créés (original, denoised, distorted).
- *birdclef.spectrograms.chunks* contient la conversion en fichiers binaires des spectrogrammes.
- *birdclef.spectrograms.files* contient les métadonnées sur les fichiers binaires en question : leur path d'origine, la taille du chunk, etc.

## Ingest vs ingest fast

Dans ingest les transformations sont faites dans des for loop, comme exemple dans le fichier `./dags/scripts/preprocess_audiofiles_to_staging.py` :
```py
...
for obj in objects:
    object_name = obj.object_name
    print(f"Traitement de l'objet : {object_name}")
    response = raw_client.get_object(raw_bucket, object_name)
    audio_data = response.read()
    response.close()
    response.release_conn()
    in_mem_file = io.BytesIO(audio_data)
    audio, sr = librosa.load(in_mem_file, sr=None)
    spec_img_bw = get_spectrogram_bw(audio, sr)
    base_name = os.path.splitext(os.path.basename(object_name))[0]
    png_name = f"{base_name}.png"
    npy_name = f"{base_name}.npy"

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
```
C'est le code sans les tests et commentaires, ici pour chaque audio file on fait 1 traitement, c'est exactement ce que l'on faisait pour le projet birdclef dans notre module de ML

Pour ce module de datalakes et pour ingest_fast, une manière d'optimiser le temps de traitement est de le faire par batch, ou en parallelisant, sachant que le traitement est un traitement fait majoritairement par la librairie `librosa` aller aussi loin que réecrire la libraire pour aller plus vite. On parallelise donc en utilisant la librairie `concurent` pour lancer les traitements en parallel.

Donc dans `preprocess_audiofiles_to_staging_fast.py` on a:
```py
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_object, obj): obj for obj in objects}
        for future in as_completed(futures):
            if future.result() is not None:
                results.append(future.result())
```

on crée notre traitement dans une fonction a call nommée "`process_object()`" qu'on appllique sur tout les objets sur plusieurs threads.

C'est la même idée pour tout les autres traitements fast.

## Instructions

Pour tout faire fonctionner chez vous il vous faut pull ce repo git: `https://github.com/Sacha-Lunae/BirdCLEF_Guenoun_ElWazzi.git` spécifiquement la branche "`ProjetDataLakes`"

normalement si tout va bien vous devriez avoir ceci:
```
ll
total 56
drwxrwxr-x 7 rami rami 4096 Mar 11 22:25 ./
drwxrwxr-x 3 rami rami 4096 Mar 11 21:50 ../
drwxrwxr-x 3 rami rami 4096 Mar 11 19:30 dags/
-rw-rw-r-- 1 rami rami 4693 Mar 11 19:30 docker-compose.yml
-rw-rw-r-- 1 rami rami  775 Feb 19 12:58 Dockerfile
drwxrwxr-x 4 rami rami 4096 Feb 17 21:04 DockerFiles/
-rw-rw-r-- 1 rami rami  311 Jan 22 16:35 .dockerignore
drwxrwxr-x 8 rami rami 4096 Mar 11 23:05 .git/
-rw-rw-r-- 1 rami rami 3108 Feb 23 17:31 .gitignore
drwxrwxrwx 7 rami rami 4096 Mar 10 21:40 logs/
drwxr-xr-x 2 root root 4096 Mar 11 22:25 plugins/
-rw-rw-r-- 1 rami rami 6168 Mar 11 23:04 README.md
```

à partir d'ici rien de plus simple que `docker-compose build` puis `docker-compose up -d` et après a peu près 5-10 min de téléchargement vous devriez avoir ces containers:
```
docker ps -a
[sudo] password for rami: 
NAMES                              CREATED          STATUS                      PORTS
airflow                            43 minutes ago   Up 43 minutes               0.0.0.0:8080->8080/tcp, [::]:8080->8080/tcp
minio-setup                        43 minutes ago   Exited (0) 43 minutes ago 
flask_api                          43 minutes ago   Up 43 minutes               0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp
mongodb                            43 minutes ago   Up 43 minutes (healthy)     0.0.0.0:27017->27017/tcp, [::]:27017->27017/tcp
postgres                           43 minutes ago   Up 43 minutes (healthy)     0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp
minio                              43 minutes ago   Up 43 minutes (healthy)     0.0.0.0:9000-9001->9000-9001/tcp, [::]:9000-9001->9000-9001/tcp

```
minio-setup est un container qui est censé exit après avoir lancé trois commandes.

à partir de la vous pouvez déjà acceder à airflow et a minio depuis localhost:8080 et localhost:9000 (ou 8081 et 9001)

les credentials de airflow sont admin, admin et minio c'est minioadmin, minioadmin

pour mongo, vous pouvez y avoir accès si vous avez mongodb compass sinon vous pouvez avoir acces au shell grâce à `docker exec -it <hash-du-container-mongodb> mongosh`

si tout va bien vous pourrez lancer les dags grâce à `curl -X POST http://localhost:8000/ingest` et `curl -X POST http://localhost:8000/ingest_fast`

vous pourrez suivre les temps d'execution depuis l'interface airflow.

En plus de ça vous avez, comme demandé, les endpoints `curl http://localhost:8000/health` et `curl http://localhost:8000/stats` en plus de `curl http://localhost:8000/compare`. On recommande de lancer ces trois requettes depuis postman ou un browser pour mieux voir les json.

Les requettes prennent du temps à retourner, souvent car elles executent les dags et attendent la fin des lancements pour retourner. Compare va lancer les deux dags et donner des stats de temps de run pour le comparatif entre ingest et ingest_fast.