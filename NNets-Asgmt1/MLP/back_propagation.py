import torch 
import torchvision
import torchvision.transforms as transform 
import numpy as np
torch.cuda.is_available()                         
'''
    Back propagation algorithm to be implemented
'''
loss_fn = torch.nn.CrossEntropyLoss()


def training_step(epoch_index,tb_writer):
    current_loss = 0.
    last_loss = 0.
    
    for i,data in enumerate(train_loader):




# create an back propagation algorithm function for cifar10 dataset```
# The function should return a dictionary containing the following keys: 
# 1. training_loss: list of training loss at each epoch
# 2. training_accuracy: list of training accuracy at each epoch
# 3. test_loss: list of test loss at each epoch
# 4. test_accuracy: list of test accuracy at each epoch
# 5. epoch_time: list of epoch time at each epoch
# 6. total_time: total time taken for training
def back_propagation():