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



Batch_size=8

dataloaders=d.load(d.training_data1,Batch_size)
train_loader=dataloaders['train']
valid_loader=dataloaders['valid']
test_loader=dataloaders['test']
print(len(train_loader)*8)
         
model = m.AlexNet ().to(m.device)
print(model)


#Defining the model hyper parameters
num_epochs =50
learning_rate = 0.0001
weight_decay =  0.01  

criterion = nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay,momentum=0.9)






#Training process begins
test_loss_list = []

train_loss_list = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}:', end = ' ')
    train_loss = 0
    test_loss=0 
    #Iterating over the training dataset in batches
    model.train()
    for i, samplebatch in enumerate(train_loader):
        images=samplebatch['image'] 
        labels=samplebatch['landmarks']
        #Extracting images and target labels for the batch being iterated
        #ici c le problème 
        
        images =  images.to(device=m.device,dtype=torch.float)
        labels= labels.type(torch.float)
        labels= labels.to(m.device)
        #print(labels)
       
  
        #Calculating the model output and the cross entropy loss
        outputs = model(images)
        loss = criterion(outputs, labels)
  
        #Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
      
    #Printing loss for each epoch
    train_loss_list.append(train_loss/(len(train_loader)*Batch_size))
    print(f"Training loss = {train_loss_list[-1]}")   
      
    #print( train_loss_list) 
    
    
    
    
       






#validation part



    test_acc=0
    model.eval()
    
    with torch.no_grad():
    
    #Iterating over the training dataset in batches
       for i, samplebatch in enumerate(valid_loader):
         images=samplebatch['image'] 
         y_true=samplebatch['landmarks']
          
         images = images.to(m.device,dtype=torch.float)
         y_true= y_true.type(torch.float)
         y_true = y_true.to(m.device)
          
        #Calculating outputs for the batch being iterated
         outputs = model(images)
         _, y_pred = torch.max(outputs.data, 1)
         loss = criterion(outputs, y_true) 
         test_loss += loss.item()
         
      
      
       test_loss_list.append(test_loss/(len(valid_loader)*Batch_size))
       print(f" validation loss = {test_loss_list[-1]}")  
       #print(test_loss_list)
       
     
#Plotting loss for all epochs  

plt.figure(3)
#plt.subplot(211)      
plt.plot(range(1,num_epochs+1), train_loss_list,'r')
plt.xlabel("Number of epochs")
plt.ylabel("Training loss")

#plt.ylim(-1,1)
#plt.show()
   
     

#plt.subplot(212)    
plt.plot(range(1,num_epochs+1), test_loss_list,'b')
plt.xlabel("Number of epochs")
plt.ylabel("testing loss")

#plt.ylim(0,0.5)
plt.show()    
      
    





