import time
import sys
sys.path.append('..')
import torch.backends.cudnn as cudnn
import torch.utils.data
from utils import *
from utils.cuda_utils import *
from loss_optimzer import loss_function,optimizer_select
from model1 import SmallMLP
from model2 import NetworkBatchNorm
from model3 import DenseMLP
from utils.data_loader_CIFAR import load_cifar10_iterators,imshow
from trainAlgorithmUtils  import adjust_learning_rate,save_checkpoint,clip_gradient,AverageMeter,ImageClassificationTraining
"""
    This training script is based on how SSD300 
    network is trained and has similarities with
    caffe repository on object detection MLPs
    
    I am using a lot of nice concepts such as
    - Decaying learning rate ~ In order to have a most accurate way to find the best (hopefully) local minimum of loss function
    - Gradient clipping ~ In order to avoid exploding gradients
    - Checkpointing ~ In order to save the model and continue training from the last saved checkpoint
    - Cuda ~ Move the data on GPU in order to speed up training process
    
"""

# Data parameters
data_folder = './Datasets'  # folder with data files

# Case Cifar10
n_classes = 10
# Case Intel 
# n_classes = 6

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
    for i, (images,labels) in enumerate(train_loader):
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
        
        start=time.time()
        
        # Print status based on iterator value and print_freq (so I get for only specific iteration print status based on modulo of print_freq)
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
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
        model = SmallMLP()
        # initialize the optimizer with biases
        biases = list()
        not_biases = list()
        
        for name,param in model.named_parameters():
            if len(param.size()) == 1:
                biases.append(param)
            else:
                not_biases.append(param)
        optimizer = optimizer_select(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],'SGD',lr=lr)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    cirterion = loss_function().to(device)
    
    # Dataloaders
        loaders_cifar = load_cifar10_iterators()
        train_loader = loaders_cifar[0]
        test_loader = loaders_cifar[1]
        val_loader = loaders_cifar[2]
        
        # Calcualte total number of epochs to train and the epochs to decay learning rate based on adaptitive learning rate
        epochs = iterations
 
        
        # Main Training LOOP
        for epoch in range(start_epoch,epochs):
            # Decay learning rate at particular epochs
            if epochs in decay_lr_at:
                adjust_learning_rate(optimizer,decay_lr_to)
            
            train_epoch(train_loader,model,criterion,optimizer,epoch)
            
        # Save Nnet as a checkpoint
        save_checkpoint(epoch,model,optimizer)
    
if __name__ = '__main__':
    main()