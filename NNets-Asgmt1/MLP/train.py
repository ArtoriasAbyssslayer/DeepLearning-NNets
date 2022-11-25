import time
import torch.backends.cudnn as cudnn
import torch.utils.data
from utils import *
from ../utils/cuda_utils import to_device
from loss_optimzer import loss_function,optimizer
from mlp1 import SmallMLP
from mlp2 import MediumMLP
from mlp3 import LargeMLP


# Data parameters
data_folder = './Datasets'  # folder with data files


n_classes = len


chackpoint = None
batch_size = 32
epochs =  100
workers = 4
print_freq = 200
lr = 1e-3 
# Slow down learning rate according to decay_lr_to factor at the end of the training 
decay_lr_at = [80,100]
decay_lr_to = 0.1
momentum = 0.9
rho  = 0.9
weight_devay = 5e-4
grad_clip = None


cudnn.benchmark = True

# Epoch training function
def train_epoch(train_loader,model,criterion,optimizer,epoch_num):
    """
        One epoch's training.
        
        :param train_loader: DataLoader for training data
        :param model: Neural Network model
        :param criterion: Loss function
        :param opimizer: optimizer
        :param epoch_num: epoch number
    """
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # backward loss counting
    
    strat = time.time()
    for (images,labels) in enumarate(train_loader):
        data_time.update(time.time() - start)
        
        # Move to default device
        images = to_device(images,device)
        labels = to_device(labels,device)
        
        # Forward propagation
        predicted_labels,prdicted_scores = model(images)
        # loss
        loss = criterion(predicted_labels,labels)
        
        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients, if we observe this is needed
        if grad_clip is not None:
            clip_gradient(optimizer,grad_clip)
        
        # Update weights
        optimizer.step()
        
        losses.update(loss.item(),images.size(0))
        batch_time.update(time.time() - start)
        
        start time.time()
        
        # Print status messages
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,,
                                                                  data_time=data_time, loss=losses))
        # delete data from gpu 
        del predicted_labels,predicted_scores,images,labeels
        
 
                                                                  
# Main function runs training for all epochs
def main():
    """
        Training for all epochs 
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at
    
    # Initialize model or load model form checkpoint (this way I can load pretrained model
    
    if checkpoint is None:
        start_epoch = 0
        model = 



if __name__ = '__main__':
    main()