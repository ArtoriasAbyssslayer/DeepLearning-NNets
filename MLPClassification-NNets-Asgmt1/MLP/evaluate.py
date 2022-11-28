import sys
sys.path.append('..')
from utils.data_loader_CIFAR import load_cifar10_iterators,imshow
import torchvision.utils
from utils.cuda_utils import *
import torch
from loss_optimizer import loss_function
import matplotlib.pyplot as plt
"""
    More comprehensive way to
    evaluate the MLP model on the test data.
"""
test_losses = []


test_acc = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be runnning on: ",device)

def eval_model(test_loader, model):
    """
    Evaluate the MLP model on the test data.

    :param test_loader: test data loader
    :param model: model
    :return: test loss and test accuracy
    """
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiters = load_cifar10_iterators()
    # Print Example for eval purposes
    test_loader = dataiters[1]
    
    # # print sample of images and eval for sample
    # images,labels = next(iter(test_loader))
    
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(32)))
    # outputs = model(images)
    # _, predicted = torch.max(outputs, 1)
    # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
    #                           for j in range(32)))
    
    # Calculate accuracy
    correct = 0
    total = 0
    temp_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            
            images = images.to(device)
            labels = labels.to(device)
           
            # Forward Inferennce
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
             # Accumulate loss
            loss = loss_function(outputs, labels)
            temp_loss += loss.item()


            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss = temp_loss / len(test_loader)
            test_losses.append(test_loss)
            acc = 100.0 * correct / total
            test_acc.append(acc)
        del images,labels,loss
        # Store the mean loss of the epoch
       
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    #Plot accuracies
    plot1 = plt.figure(1)
    plt.plot(test_acc, '-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Test'])
    plt.title('Test Accuracy')

    # Plot losses
    plot2 = plt.figure(2)
    plt.plot(test_losses, '-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Test'])
    plt.title('Test Losses')

    return correct_pred, total_pred,test_acc,test_losses


