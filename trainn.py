import os
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import model_simple as m
import datasetprojet as d
from get import get_params
from hyperband import Hyperband

def main(arg,num_epochs =50):
  
   Batch_size =args.batchsize# 8 par défaut
   learning_rate=args.learning_rate        
					
        
           # 0.0001
   weight_decay = args.weight_decay

   dataloaders=d.load(d.training_data1,Batch_size)
   train_loader=dataloaders['train']
   valid_loader=dataloaders['valid']
   test_loader=dataloaders['test']
   print(len(train_loader)*8)
   #model=m.ResNet50(num_classes=1).to(m.device)      
   model = m.AlexNet ().to(m.device)
   model.apply(model.init_weights)
   print(model)


#Defining the model hyper parameters
   
   #learning_rate = 0.0001
   #weight_decay =  0.01
 

   criterion = nn.MSELoss(reduction = 'sum')
   optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
   #optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)






#Training process begins
   validation_loss_list = []

   train_loss_list = []
   for epoch in range(num_epochs):
       print(f'Epoch {epoch+1}/{num_epochs}:', end = ' ')
       train_loss = 0
       valid_loss=0 
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
            valid_loss += loss.item()
         
      
      
          validation_loss_list.append(valid_loss/(len(valid_loader)*Batch_size))
          print(f" validation loss = {validation_loss_list[-1]}") 
   best_training_loss=min(train_loss_list )
   best_validation_loss=min(validation_loss_list)      
   print('best training loss',best_training_loss) 
   print('best validation loss',best_validation_loss)
         #print(test_loss_list)
       
   train_loss_list_plot=train_loss_list[1:]
   validation_loss_list_plot=validation_loss_list[1:]
  #Plotting loss for all epochs  

   plt.figure(3)
  #plt.subplot(211)      
   plt.plot(range(2,num_epochs+1), train_loss_list_plot,'r',label="training loss")
   plt.xlabel("Number of epochs")
   

   #plt.ylim(-1,1)
   #plt.show()
   
     

   #plt.subplot(212)    
   plt.plot(range(2,num_epochs+1), validation_loss_list_plot,'b',label="validation loss")
   plt.xlabel("Number of epochs")
   plt.ylabel("MSE")

   #plt.ylim(0,0.5)
   #plt.show()   
   plt.legend()
   plt.savefig('plot.png') 
   #plt.close()



  ### testing part our model with test loader created 
   test_loss_final=0.0
   with torch.no_grad():
    
      #Iterating over the training dataset in batches
          for i, samplebatch in enumerate(test_loader):
            images=samplebatch['image'] 
            y_true=samplebatch['landmarks']
          
            images = images.to(m.device,dtype=torch.float)
            y_true= y_true.type(torch.float)
            y_true = y_true.to(m.device)
          
          #Calculating outputs for the batch being iterated
            outputs = model(images)
        
            loss = criterion(outputs, y_true) 
            test_loss_final += loss.item()
         
      
      
          test_error_final=test_loss_final/(len(test_loader)*Batch_size)
          print(f" testing loss = {test_error_final}")  
          #print(test_loss_list)
   return {"best train loss":best_training_loss,"best_val_loss":best_validation_loss}
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train a line detector')
  parser.add_argument('--batchsize', help='Batch size', default= 8, type=int)
  parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=0.0001, type=float)#same here
  parser.add_argument('-wd', '--weight_decay', help='Weight_decay', default=0.001, type=float) #i need them just for training after finding the best paramaters by hyperband algorithm
  parser.add_argument('--hyp', help='Test hyperparameters', default=0, type=int)
  parser.add_argument('--num_epochs', help='Number of epoch', default= 50, type=int)
  args = parser.parse_args()
  
  if (args.hyp==0):
    hyperpar= {}
        
    hyperpar['batchsize'] = args.batchsize
    hyperpar['learning_rate'] = args.learning_rate
        
    hyperpar["weight_decay"] = args.weight_decay
        
    main(hyperpar,args.num_epochs)
  
  elif (args.hyp==1):
    hyp=Hyperband(args,get_params,main)
    hyp.run( dry_run = False, hb_result_file = "/home/student/ross/hb_result.json", hb_best_result_file = "/home/student/ross/hb_best_result.json" )
     
  
#car on a argument test dont on n'a pas besoin
    
        
  
    
    

     

    


