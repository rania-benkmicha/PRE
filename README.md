# PRE

This project is about learning to estimate terrain traversability from vision for a mobile robot using a self-supervised approach.


# Code overview

- `bagfiles/` contains the raw data as bagfiles and some helper scripts.
  - `filter_bag.sh` contains the command used to extract the sample bag from the full data.
  - `sample_bag.bag` is a short sample file, the full dataset (`rania_2022-07-01-11-40-52.bag`) is available at https://drive.google.com/drive/folders/1aEfvWY1DxogPogli_FlXV0OhHR5cS7g-?usp=sharing. 
  - `rosbag_record_topic_list.txt` is the list of topics that should be recorded on the robot. 

- `datasets/` contains the dataset created from bagfiles processing

- `create_dataset.py` will process a bag file to create a self-supervised dataset

- `show_dataset.py` will create a collage of worst and best images from dataset

- `train_test.py` will lanch training and test of the model

- `generate_rand_params.py` generates random hyperparameters configurations

- `hyperband.py` definitions for hyperband algorithm

- `modele_simple.py` contains the description of the neural networks

- `loader.py` defines dataloaders and data augmentation

# Code usage

Start by creating the dataset from the bag files, e.g.:

`python create_dataset.py bagfiles/sample_bag.bag`

Then start training, e.g.:

`python train_test.py --batchsize 8 --learning_rate 0.001 --weight_decay 0.002 --hyp 0 --modelnetwork AlexNet`

where:

- `weight decay` is the weight of L2 regularization.

- `hyp` should be 0 to train with hyper-parameters given as parameters, or 1 to run hyperband

- `modelnetwork` can be "AlexNet" or "ResNet".

