import os

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import model_simple as m
import datasetprojet as d




train_loader=d.load(d.training_data1,8)
#train_loader=dataloaders['train']
#valid_loader=dataloaders['valid']
#test_loader=dataloaders['test']

         
model = m.AlexNet ().to(m.device)
print(model)


#Defining the model hyper parameters
num_epochs = 10
learning_rate = 0.1
weight_decay = 0.001

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)







#Training process begins
train_loss_list = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}:', end = ' ')
    train_loss = 0
      
    #Iterating over the training dataset in batches
    model.train()
    for i, samplebatch in enumerate(train_loader):
        images=samplebatch['image'] 
        labels=samplebatch['landmarks']
        #Extracting images and target labels for the batch being iterated
        images =  images.to(device=m.device,dtype=torch.float)
        labels= labels.type(torch.LongTensor)
        labels= labels.to(m.device)
       
  
        #Calculating the model output and the cross entropy loss
        outputs = model(images)
        loss = criterion(outputs, labels)
  
        #Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
      
    #Printing loss for each epoch
    train_loss_list.append(train_loss/len(train_loader))
    print(f"Training loss = {train_loss_list[-1]}")   
      
    #print( train_loss_list) 
    
    
    
    
       
#Plotting loss for all epochs
plt.plot(range(1,num_epochs+1), train_loss_list)
plt.xlabel("Number of epochs")
plt.ylabel("Training loss")







