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
    
    for i,data in enumerate(train_loader)




