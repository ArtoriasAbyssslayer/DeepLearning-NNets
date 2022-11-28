import time
import os
import sys
import torch
import torch.nn
sys.path.append('..')
import torch.backends.cudnn as cudnn
import torch.utils.data
from utils import *
from utils.cuda_utils import *
from loss_optimizer import loss_function,optimizer_select
from utils.data_loader_CIFAR import load_cifar10_iterators,load_cifar10_dataset
from trainAlgorithmUtils  import adjust_learning_rate,save_checkpoint,clip_gradient,AverageMeter,save_model
from model1 import SmallMLP
from model2 import NetworkBatchNorm
from model3 import DenseMLP
import matplotlib.pyplot as plt
from evaluate import eval_model
from tqdm import tqdm
import ssl
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
iterations = 120000  # number of iterations to train
checkpoint = None  # path to model checkpoint, None if none
batch_size = 32
epochs =  125
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

train_losses = []
train_acc = []
cudnn.benchmark = True
classes_cifar = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_intel = ('buildings', 'forest', 'glacier', 'mountain', 'sea', 'street')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    n_samples = 0
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # backward loss counting
    running_accuracies = AverageMeter() # accuracy counting
    train_loader = DeviceDataLoader(train_loader, device)
    start = time.time()
    for i,(images,labels) in enumerate(train_loader):
        data_time.update(time.time() - start)
        
        # Move to default device
        images = to_device(images,device)
        # labels = torch.nn.functional.one_hot(labels, num_classes=len(classes_cifar))
        # print(labels)
        labels = to_device(labels,device)
        
        # Forward propagation
        outputs = model(images)
        # print(outputs)
        _, predicted = torch.max(outputs, 1)
        
        
        # loss
        loss = criterion(outputs,labels)
        accuracy = 0.0
        accuracy += (predicted == labels).sum().item()
        n_samples += labels.shape[0]
        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients, if we observe this is needed
        if grad_clip is not None:
            clip_gradient(optimizer,grad_clip)
        
        # Update weights
        optimizer.step()
        start=time.time()
        # Update loss,accuracy,batchtime
        losses.update(loss.item(),images.size(0))
        batch_time.update(time.time() - start)
        # compute and print the average running_acc for this epoch when tested all 10000 test images
        running_accuracies.update(accuracy, images.size(0))
        train_losses.append(losses.avg)
        train_acc.append(100*accuracy/n_samples)
        # Print status based on iterator value and print_freq (so I get for only specific iteration print status based on modulo of print_freq)
        if i % print_freq == 0:
            print('Predicted: ', ' '.join('%5s' % classes_cifar[predicted[j]]
                            for j in range(1)))
        print('Epoch: [{0}][{1}/{2}]\t'
                'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Train Accuracy{accuracy.val:.3f}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch_num, i, len(train_loader),
                                                                batch_time=batch_time,
                                                                data_time=data_time, accuracy=running_accuracies, loss=losses))
    del predicted,loss,outputs,images,labels,train_loader 
        
    
    
    
   
        
        
                                                             
# Main function runs training for all epochs
def mainTraining(model):
    """
        Training for all epochs 
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be runnning on: ",device)
    # Initialize model or load model form checkpoint (this way I can load pretrained model
    
    if checkpoint is None:
        start_epoch = 0
        if model == 'SmallMLP':
            model = SmallMLP()
        elif model == 'NetworkBatchNorm':
            model = NetworkBatchNorm()
        elif model == 'DenseMLP':
            model = DenseMLP(activation="relu")
        # initialize the optimizer with biases
        biases = list()
        not_biases = list()
        
        for name,param in model.named_parameters():
            if len(param.size()) == 1:
                biases.append(param)
            else:
                not_biases.append(param)
        optimizer = optimizer_select(net_params=[{'params': biases}, {'params': not_biases}],type='SGD',lr=lr)
        
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    
    # Dataloaders
    train_loader,test_loader,val_loader = load_cifar10_iterators(data_folder,batch_size,workers)  
    train_dataset,_ = load_cifar10_dataset()  
    # Calcualte total number of epochs to train and the epochs to decay learning rate based on adaptitive learning rate
    epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]
    
    # Main Training LOOP
    for epoch in tqdm(range(start_epoch,epochs)):
        # Decay learning rate at particular epochs
        if epochs in decay_lr_at:
            adjust_learning_rate(optimizer,decay_lr_to)
        
        train_epoch(train_loader,model=model,criterion=loss_function,optimizer=optimizer,epoch_num=epoch)
        
        # Save Nnet as a checkpoint
        save_checkpoint(epoch,model,optimizer)
    save_model(model,sys.argv[1])
    # Plot training curves
    print('Loss and Accuracy Curves')
    # Evaluate the model on test set 
    _,_,test_acc,test_losses=eval_model(test_loader=test_loader,model=model)
    # Plot accuracies
    plot1 = plt.figure(1)
    plt.plot(train_acc, '-o')
    plt.plot(test_acc, '-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Test'])
    plt.title('Train vs Test Accuracy')
    plt.show()

    # Plot losses
    plot2 = plt.figure(2)
    plt.plot(train_losses, '-o')
    plt.plot(test_losses, '-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train', 'Test'])
    plt.title('Train vs Test Losses')
    plt.show()

if __name__ == '__main__':
    mainTraining(model=sys.argv[1])
    