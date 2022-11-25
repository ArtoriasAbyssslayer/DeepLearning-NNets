import torch 
import torch.nn as nn
import torch.nn.functional as F
from loss_optimzer import loss_function
torch.cuda.is_available()                         
'''
    Back propagation algorithm is implemented in fit function
    after forward pass is done and we have the prediction errors as
    tought in class.
'''


class ImageClassificationTraining(nn.Module):
    # Define training step algorithm that uses batch training and cycles through training set
    def training_step(self,batch_index,batch):
        images,labels = batch
        out = self(images)   # Generate predictions with self function from nn.Module
        current_loss = loss_function(images,labels) # Calculate prediction loss 
        return current_loss
        
        for i,data in enumerate(train_loader):
    # Validation step is
    def validation_step(self,batch):
        images,labels = batch
        out = self(images)
        loss = 



# create an back propagation algorithm function for cifar10 dataset```
# The function should return a dictionary containing the following keys: 
# 1. training_loss: list of training loss at each epoch
# 2. training_accuracy: list of training accuracy at each epoch
# 3. test_loss: list of test loss at each epoch
# 4. test_accuracy: list of test accuracy at each epoch
# 5. epoch_time: list of epoch time at each epoch
# 6. total_time: total time taken for training
def backward_loss()