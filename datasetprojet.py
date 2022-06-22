#preparation de l'environnement

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#plt.ion()   # interactive mode


#Création de classe


class DataProject(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        
       
       
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.data_frame.iloc[idx, 1:]#type numpy.float64
        landmarks = np.array(landmarks)
        #landmarks=float(landmarks)
        
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
        
    #pour le test de size
    def affichage(self):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[5, 0])
        image = io.imread(img_name)
        output=type(image)
        tupleee=image.shape
        print(tupleee)
        
        print(type(image))
        
        




#Data vizualization

def data_plot(dataproject,cols,rows):
  
  dataproject.affichage()
  figure = plt.figure(figsize=(10, 10))
  
  for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(dataproject), size=(1,)).item()
    sample = dataproject[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(sample['landmarks'])
    
    plt.axis("off")
    plt.imshow(sample['image'].squeeze())
  plt.show()
  
 
 #loading data

 
## transformation des images:




class ToTensor(object):
    

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': landmarks}



        
#loading DataSet after transformation 


training_data1 = DataProject(
    csv_file='/home/student/ross/Bags_Data_2022-04-14-17-40-50/_imu_data.csv',root_dir='/home/student/ross/Bags_Data_2022-04-14-17-40-50/_zed_node_left_image_rect_color',transform=ToTensor()

)


for i in range(len(training_data1)):
    sample = training_data1[i]
data_plot(training_data1,3,3)
    

#data_plot(training_data1,3,3)




#dataloader = DataLoader(training_data, batch_size=4,shuffle=True, num_workers=0)



