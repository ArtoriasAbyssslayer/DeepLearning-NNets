from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from knn_generic import knn_custom,mode
import sys
sys.path.append('../../NNets-Asgmt1/')
from utils.data_loader_CIFAR import load_cifar10_iterators,imshow
from utils.data_loader_Intel import load_Intel 
from sklearn.metrics import classification_report,accuracy_score
# from metric import accuracy
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torchvision.utils
import time 
from tqdm import tqdm 
# define classes in a tuple
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# define images shape 
# img_rows, img_cols = 32, 32
# input_shape = (img_rows, img_cols, 3)

loaders = load_cifar10_iterators()
train_loader = loaders[0]
test_loader = loaders[1]
val_loader = loaders[2]
# Print Some Data for observation purposes DataLoader Batchsize = 256 no take only 16th row
for images,classes in train_loader:
    print('images.shape:', images.shape)
    print('classes.shape', classes.shape)
    plt.axis('off')
    # without normalization
    imshow(torchvision.utils.make_grid(images,n_row=10))
    # plt.imshow(torchvision.utils.make_grid(images,nrow=32).permute((1,2,0)))
    break
# The same goes for images in other sets

# Trying to record accuracy_score of sklearn.KNeighborsClassifier and custom knn with 1 to 3 Neighbors
k_range = range(1,3)
scores = {}
scores2= {}
times = {}
times2 = {}
scores_list = []
scores2_list = []
times_list = []
times2_list = []
for k in tqdm(k_range):
    for x_train,y_train in train_loader:
        for x_test,y_test in test_loader:
            x_train = np.reshape(x_train, (x_train.shape[0], -1))
            x_test = np.reshape(x_test, (x_test.shape[0], -1))
            start1 = time.time()
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train,y_train)
            y_pred=knn.predict(x_test)
            end1 = time.time()
            print(classification_report(y_test,y_pred))
            times[k] = end1-start1
            times_list.append(times[k])
            scores[k] = accuracy_score(y_test,y_pred)
            scores_list.append(scores[k])
            start2 = time.time()
            x_pred2,y_pred2 = knn_custom(k,x_test,x_train,mode)
            end2 = time.time()
            times2[k] = end2-start2
            print(classification_report(y_test,y_pred2))
            times2_list.append(times2[k])
            scores2[k] = accuracy_score(y_test,y_pred2)
            scores2_list.append(scores2[k])
# Classification report 
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
best_k = np.argmax(scores_list)
best_k_generic = np.argmax(scores2_list)

#after running the code above, assign to n_neighbors the best performing value
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(x_train,y_train)
y_pred = knn.predict(x_test)
cmat = confusion_matrix(y_test,y_pred,labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cmat,display_labels=classes)
disp.plot(cmap=plt.cm.Reds)
knn_best_generic = knn_custom(best_k_generic,y_test,x_test,mode)
cmat2 = confusion_matrix(y_test,knn_best_generic.y_pred,labels=classes)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cmat2,display_labels=classes)
disp.plot(cmap=plt.cm.Blues)