from urllib.request import urlretrieve
import numpy 
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch 
 # TODO FINISH THE LOADERS FOR MNIST
'''Data Loading Utility Functions'''


'''Data Masks'''

"""
    Create Gaussian Blur Transformer in order to 
    test denoising capabilities of AE models
"""

class GuassianBlurTransform(object):
	def __init__(self,kernel_size,sigma):
		self.kernel_size =  kernel_size
		self.sigma = sigma
	def __call__(self,x):
		guassian_blur = transforms.GaussianBlur(kernel_size,sigma)
		blurred_img = guassian_blur(x)
		return blurred_img

"""
    Create a Threshold Transformer in order to 
    Nomralize images so that Letter Details Preserved (above 0.5 brightness -> 1 else 0)
"""

class ThresholdMaskTransform(object):
	def __init__(self, thr):
		self.thr = thr 
	def __call__(self,x):
		return (x > self.thr).to(x.dtype('float32')) 

'''Label Transform'''
labels=['0','1','2','3','4','5','6','7','8','9']
class OneHotLabelTransform(object):
    def __init__(self,n_classes):
        self.n_classes = n_classes
    def __call__(self,labels):
        one_hot_labels =  torch.zeros(len(labels),self.n_classes)
        for label in labels:
            one_hot_labels = F.one_hot(label,self.n_classes)
        return one_hot_labels.float()



def load_mnist(batch_size=32, masking=None, one_hot_labels=False,workers=4):
    """
        Mnist Load function
        :param batch_size: batch size
        :param masking: type of masking to apply to images
        :param one_hot_labels: whether to use one hot labels or not
        :param workers: parallel download threads
        :return: train_DataLoader,test_DataLoader
    """
	# data preporc
    if masking == "guassian_maskin":
        data_Transformer = transforms.Compose([
            ToTensor(), 
            GuassianBlurTransform(kernel_size=3, sigma=0.001)
            ])
    elif masking == "bin_masking":
        data_Transformer = transforms.Compose([
            ToTensor(), 
            ThresholdMaskTransform(thr=0.5)
            ])
    else:
        data_Transformer = ToTensor()

    # label preproc
    if one_hot_labels:
        label_Transformer = transforms.Compose([
            OneHotLabelTransform(n_classes=10,labels=labels)
            ])

    else:
        label_Transformer = None

    train_data =  datasets.MNIST(root='data',
								   train=True,
								   download=True,
								   transform=data_Transformer,
                                   target_transform=label_Transformer)
    test_data = datasets.MNIST(root='data',
							   train=False,
							   download=True,
							   transform=data_Transformer,
                               target_transform=label_Transformer)

    train_DataLoader = DataLoader(train_data,batch_size=batch_size,shuffle=True, num_workers=workers, pin_memory=True)
    test_DataLoader = DataLoader(test_data,batch_size=batch_size,num_workers=workers, pin_memory=True)
    return train_DataLoader,test_DataLoader



'''Training Utilities'''



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



'''Save Model Util Functions'''

import os 


savepath = './saved_models/'
def save_pca_model(model,model_name):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    torch.save(model.state_dict(), savepath+model_name)
    print('Model saved to '+savepath+model_name)
    return

def load_pca_model(model,model_name):
    model.load_state_dict(torch.load(savepath+model_name))
    print('Model loaded from '+savepath+model_name)
    return

def save_model(model,optimizer,save_name):
    if not os.path.exists(savepath):   
        os.mkdir(savepath)
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': loss,
            }, savepath+save_name)
    print("-Model Saved ....!")


    
def save_checkpoint(epoch,min_loss,model, optimizer, suffix=False,save_name=None):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    :param suffix: whether to append epoch number to filename
    """
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             'test_loss': min_loss
             }
    if save_name == None:
        filename = savepath+'checkpoint_{}.ptr'.format(model.__class__.__name__)
    else:
        filename = savepath+'checkpoint_{}.ptr'.format(model.__class__.__name__)
    if suffix:
        filename = savepath+'checkpoint_epoch_{}_{}.ptr'.format(epoch,model.__class__.__name__)
    torch.save(state, filename)
    print("-Model checkpoint Saved ....!")