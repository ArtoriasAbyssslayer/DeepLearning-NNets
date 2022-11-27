import sys
sys.path.append("../")
from utils.data_loader_CIFAR import load_cifar10_iterators
from utils.cuda_utils import DeviceDataLoader,  to_device
from torch.autograd import Variable

def train(num_epochs,model):
    best_accuracy = 0.0
    
    # Define execution to CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be runnning on: ",device)
    # Convert model parameters and buffers to CPU or CUDA tensors
    model.to(device)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        
        for i, (images,labels) in enumerate(train_loader,0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            
            # select optimizer and loss function
            