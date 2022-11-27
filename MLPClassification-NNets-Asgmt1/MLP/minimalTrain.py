import sys
sys.path.append("../")
from utils.data_loader_CIFAR import load_cifar10_iterators
from utils.cuda_utils import DeviceDataLoader,  to_device
import torch
from torch.autograd import Variable
from loss_optimzer import optimizer_select, loss_select
from trainAlgorithmUtils import *
def minimal_train(num_epochs,model):
    """
        Minimalistic approach to train the model.
        Optimizer selected is ADAM (because it utilizes mini-batch training)
        loss function is CrossEntropyLoss
        Adaptive optimizer is not applied neither gradient clipping or other fancy 
        optimization methods that were applied in train.py
    """
    loss_func = loss_select()
    best_accuracy = 0.0
    optimizer = optimizer_select(model,type='Adam',lr=0.001)
    # buffers that will be used to store the loss and accuracy of the model
    train_losses = []
    valid_losses = []
    # Define execution to CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be runnning on: ",device)
    # Convert model parameters and buffers to CPU or CUDA tensors
    model.to(device)
    train_loader,_ ,val_loader = load_cifar10_iterators()
    for epoch in range(1,num_epochs+1):
        running_loss = 0.0
        running_acc = 0.0
        
        for i, (images,labels) in enumerate(train_loader,0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(images)
            # calculate the batch loss
            loss = loss_func(output,labels)
            # backward-pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # performa a single optimization step (parameter update)
            optimizer.step()
            # update training loss            
            running_loss += loss.item() * images.size(0)
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0
            # compute and print the average accuracy for this epoch when tested all 10000 test images
            accuracy = testAccuracy()
            print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
            if accuracy > best_accuracy:
                save_checkpoint(model,optimizer,epoch,accuracy)
                save_model()
                best_accuracy = accuracy
            
            # another model evaluation method
            model.eval()
            for image,label in val_loader:
                
                image = image.to(device)
                label = label.to(device)
                
                output = model(image)
                
                loss = loss_func(output,label)
                
                # update average validation loss
                valid_loss += loss.item()*image.size(0)
            
            # calculate average losses
            train_loss = train_loss/len(train_loader.sampler)
            validation_loss = valid_loss/len(val_loader.sampler)
            train_losses.append(train_loss)
            valid_losses.append(validation_loss)
            
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch,train_loss,validation_loss))
            return train_losses,valid_losses                                                                                        