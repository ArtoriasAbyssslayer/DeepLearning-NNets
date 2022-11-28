import torch 
import torch.nn as nn
import torch.nn.functional as F
from loss_optimizer import loss_function
from utils.data_loader_CIFAR import load_cifar10_iterators
torch.cuda.is_available()                         
"""
    This file contains some utility functions
    needed for training and testing the network.

    Also contains an ImageClassificationBaseClass
    That is the way that was the first implemented 
    training and testing class of the neural network.
    
    Edit: This class is not used anymore. Instead,
    I have a minimilastic train approach in minimalTrain.py
    and a more comprehensive(regarding optimization tricks/methods)
    train approach in train.py.
    
"""

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))
    # Define a more minimalistic version of evaluation
def testAccuracy(model,testloader):
    model.eval()
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiters = load_cifar10_iterators()

    test_loader = dataiters[1]
    
    accuracy = 0.0
    total = 0.0
    with torch.no_grad(): # No need to calculate gradients for testing purposes
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    Got it from SSD implementation 
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def save_model(model):
    path = "./saved_models/"
    torch.save(model.state_dict(), path)


    
def save_checkpoint(epoch, model, optimizer, suffix=False):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    :param suffix: whether to append epoch number to filename
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_mlp_classification.pth.tar'
    if suffix:
        filename = 'checkpoint_cifar10_epoch{}.pth.tar'.format(epoch)
    torch.save(state, filename)