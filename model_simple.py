import os

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

#selecting the device to train our model onto, i.e., CPU or a GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")





#essai de modèle AlexNet sans régularisation

class AlexNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
      
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            #input 3*225*225 output 64 *61*61
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3, stride=2),
            #input 64*61*61 output 64*30*30
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # target output size of 6*6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            #nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,1),
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
       
        x = self.avgpool(x)
        #print(x.shape)
        x = torch.flatten(x, 1)#same as x.view
        #print(x.shape)
        x = self.classifier(x)
        return x
        
        
                

        
        
        
    
