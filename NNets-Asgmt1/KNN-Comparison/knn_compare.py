from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from knn_generic import knn,euclidean_distance,mode
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append('../../NNets-Asgmt1/')
from utils.data_loader_CIFAR import load_cifar10_iterators,imshow
from utils.data_loader_Intel import load_Intel 
from sklearn import metrics
import math
import time 
from tqdm import tqdm 
# define classes in a tuple
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# define images shape 
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)

# Invoke the above function to get our data.
# x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()
trainloader,testloader = load_cifar10_iterators()
traindataiter = iter(trainloader)
x_train,y_train =  next(traindataiter)


testdataiter = iter(testloader)
y_test,x_test = next(testdataiter)
# show images
imshow(torchvision.utils.make_grid(images))
# print labels batch_size = 4
batch_size = 4
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
print('Train data shape: ', x_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', x_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)

# Trying to record accuracy of sklearn.KNeighborsClassifier and custom knn with 1 to 3 Neighbors
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
    start1 = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(x_test)
    end1 = time.time()
    print(metrics.classification_report(y_test,y_pred,classes))
    times[k] = end1-start1
    times_list.append(times[k])
    scores[k] = metrics.accuracy_score(y_train,y_pred)
    scores_list.append(scores[k])
    start2 = time.time()
    y_test2,y_pred2 = knn(k,y_train,x_train,euclidean_distance,mode)
    end2 = time.time()
    times2[k] = end2-start2
    times2_list.append(times2[k])
    scores2[k] = metrics.accuracy_score(y_test2,y_pred2)
    scores2_list.append(scores2[k])
# Classification report 


#after running the code above, assign to n_neighbors the best performing value
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
y_pred= knn.predict(x_test)
knn2 = knn(1,y_train,x_train,euclidean_distance,)
