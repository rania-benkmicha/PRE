# PRE

This project is about learning to estimate terrain traversability from vision for a mobile robot using a self-supervised approach.


# Code overview

- `bagfiles/` contains the raw data as bagfiles. A sample file is included, the full dataset (`rania_2022-07-01-11-40-52.bag`) is available at https://drive.google.com/drive/folders/1aEfvWY1DxogPogli_FlXV0OhHR5cS7g-?usp=sharing. `filter_bag.sh` contains the command used to extract the sample bag from the full data.


train_test.py: c'est le fichier qui est responsable de l'entrainement et de test de modèle.

get.py: c'est le fichier qui permet de préparer les configurations de modèle.

hyperband.py: Il s'agit d'une implémentation de l'algorithme d'optimisation des hyper-paramètres "hyperband".

modele_simple.py: c'est le fichier qui contient l'implémentation des structures de réseaux de neurones.

datasetprojet.py: c'est le fichier responsable de chargement et  traitement de la base de données.

read_bags0.py: c'est le script ROS qui permet de générer la base de données du robot.

imu.csv: c'est la base de données sur laquelle nous travaillons.

rosbag_record_topic_list_rania.txt: c'est la liste de topics utilisés pour extraire les informations de robot.



# Pour exécuter le code , il suffit de taper la ligne de code suivante:

python train_test.py --batchsize 8 --learning_rate 0.001 --weight_decay 0.002 --hyp 0 --modelnetwork AlexNet
tel que:

batchsize: c'est la taille de l'échantillon.

learning rate: c'est le pas de l'algorithme d'optimisation.

weight decay: c'est le paramètre de pénalité pour la régularisation L2.

hyp: S'il est égal 0 signifie qu'on est en train de vérifier le modèle avec nos choix de paramètres.S'il est égale à 1 ,cela signifie q'on lance l'algorithme d'optimisation de paramètres "hyperband".

modelnetwork: il repésente la nature de modèle, nous avons le choix entre "AlexNet" et "ResNet".




