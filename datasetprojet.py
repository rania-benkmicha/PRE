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
       
        landmarks=float(landmarks)
        
        sample = {'image': image, 'landmarks': landmarks}
#hne 5ater ye5dedh dict

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
  #figure = plt.figure(figsize=(200, 200))
  #figure2=plt.figure()
  for i in range(1, cols * rows + 1):
    #sample_idx = torch.randint(len(dataproject), size=(1,)).item()
    plt.figure(1)
    plt.subplot(rows,cols,i)
    sample = dataproject[i]
    #figure.add_subplot(rows, cols, i)
    
    #plt.tight_layout()
    plt.title(sample['landmarks'])
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.axis("off")
    plt.imshow(sample['image'].squeeze())
    
   
    plt.figure(2)
    plt.subplot(rows,cols,i)
    plt.hist(np.array(sample['image']).ravel(), bins=50, density=True)
    plt.xlabel("pixel values")
    plt.ylabel("relative frequency")
    plt.title("distribution of pixels")
    plt.subplots_adjust(wspace=1, hspace=1)
    
  plt.show()
  
 
 
 
 #loading data

#data_plot(training_data,5,5)
 
## transformation des images:


###rescaling image's size




class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        #if isinstance(self.output_size, int):
            #if h > w:
                #new_h, new_w = self.output_size * h / w, self.output_size
            #else:
                #new_h, new_w = self.output_size, self.output_size * w / h
        #else:
        new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        #landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}




class ToTensor(object):
    

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        ######(a voir si'il y aura un problème)
        return {'image': torch.from_numpy(image),
                'landmarks': landmarks}


      




training_data = DataProject(
    csv_file='/home/student/ross/Bags_Data_2022-04-14-17-40-50/_imu_data.csv',root_dir='/home/student/ross/Bags_Data_2022-04-14-17-40-50/_zed_node_left_image_rect_color'

)

data_plot(training_data,5,5)



#normalize images for regression problems

training_data = DataProject(
    csv_file='/home/student/ross/Bags_Data_2022-04-14-17-40-50/_imu_data.csv',root_dir='/home/student/ross/Bags_Data_2022-04-14-17-40-50/_zed_node_left_image_rect_color'

)
  
def calcul_moy_std(training_data):  
  
  sumr=0
  sumg=0
  sumb=0
  stdr=0
  stdb=0
  stdg=0
  loader = DataLoader(training_data, batch_size=len(training_data), num_workers=0)
  data = next(iter(loader))
  m=(data ['image'][0]).numpy()
#je suis pas sur de mean axis

  h,w,c=m.shape
  for i in data['image']:
    m=i.numpy()
    r=m[:,:,0] 
    g=m[:,:,1] 
    b=m[:,:,2] 
    sumr=sumr+r.sum()
    sumg=sumg+g.sum()
    sumb=sumb+b.sum()
  print(h,w)
  
  
  
  meanr=sumr/(h*w*len(training_data))
  meang=sumg/(h*w*len(training_data))
  meanb=sumb/(h*w*len(training_data))
  mean=(meanr,meang,meanb)
  print('mean')
  print(meanr,meang,meanb)
  for i in data['image']:
    m=i.numpy()
    r=m[:,:,0] 
    g=m[:,:,1] 
    b=m[:,:,2]   
    stdr=stdr+((r - meanr)**2).sum()
    stdg=stdg+((g - meang)**2).sum()
    stdb=stdb+((b - meanb)**2).sum()
  stddr=np.sqrt(stdr/(h*w*len(training_data)))
  stddb=np.sqrt(stdb/(h*w*len(training_data)))
  stddg=np.sqrt(stdg/(h*w*len(training_data)))
  std=(stddr,stddg,stddb)

  return(mean,std)


class Normalize(object):
 
    def __init__(self, mean, std):
        self.std = std
        self.mean = mean
        
    def __call__(self, sample):
        
         #tr=transforms.Normalize(self.mean,self.std)
        #return tr(sample['image'])
        sample['image'][:,:,0]=(sample['image'][:,:,0]-self.mean[0])/self.std[0]
        sample['image'][:,:,1]=(sample['image'][:,:,1]-self.mean[1])/self.std[1]
        sample['image'][:,:,2]=(sample['image'][:,:,2]-self.mean[2])/self.std[2]
        return sample


        
#loading DataSet after transformation 


t=calcul_moy_std(training_data)
print('le coue',t)

data_transforms = transforms.Compose([
    #Normalize(t[0],t[1]),
    # Apply histogram equalization
    Rescale(255),
    
    ToTensor()
    
    
     # Add channel dimension to be able to apply convolutions
    
])



training_data1 = DataProject(
    csv_file='/home/student/ross/Bags_Data_2022-04-14-17-40-50/_imu_data.csv',root_dir='/home/student/ross/Bags_Data_2022-04-14-17-40-50/_zed_node_left_image_rect_color',transform=data_transforms
)


#data_plot(training_data1,7,7)
print('hola')
#for i in range(len(training_data1)):
    #sample = training_data1[i]
    #print(sample['image'].shape)
    #print(sample['landmarks'])

    




# loading data in an iterator dataloader
#shuffle= True pour le choix aleatoire
dataloader = DataLoader(training_data1, batch_size=8,shuffle=True, num_workers=0)

#visualizing a batch

def show_landmarks_batch(sample_batched):
    
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))#from tensor to ndarray

    #for i in range(batch_size):
        
        
        #plt.title('Batch from dataloader')








for i_batch, sample_batched in enumerate(dataloader):
    #print(i_batch, sample_batched['image'].size(),
          #sample_batched['landmarks'].size())

    
    if i_batch == 1:
        plt.figure(1)
        show_landmarks_batch(sample_batched)
        print(sample_batched ['image'].size())
        plt.title((sample_batched['landmarks']).tolist())
        plt.axis('off')
        plt.ioff()
        plt.show()

        break
