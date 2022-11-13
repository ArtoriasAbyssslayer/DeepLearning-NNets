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
import seaborn as sns
# define classes in a tuple
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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
times = {}
scores_list = []
times_list = []

for k in tqdm(k_range):
    for x_train,y_train in train_loader:
            testiter = iter(test_loader)
            x_test,y_test = next(testiter)
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
# Classification report 
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
best_k = np.argmax(scores_list)

most_time = np.argmax(times_list)
print("Slowest knn classifier took : {}seconds".format(times_list[most_time]))
#after running the code above, assign to n_neighbors the best performing value
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(x_train,y_train)
y_pred = knn.predict(x_test)
cmat = confusion_matrix(y_test,y_pred)
plt.figure(0)
disp = ConfusionMatrixDisplay(confusion_matrix=cmat,display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.figure(3)
sns.heatmap(cmat, annot=True)
