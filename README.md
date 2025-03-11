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

## Instructions