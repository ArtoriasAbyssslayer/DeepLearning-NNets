
import sys
sys.path.append('..')
from sklearn import svm
import time
import numpy as np
import matplotlib.pyplot as plt
from auxiliary_funcs.data_preprocessing import cifar10_load,data_preproc,train_test_split,reduce_dataset_size
from auxiliary_funcs.imshow_denormalized import imshow
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
classesName = ['airplane', 'car', 
               'bird', 'cat', 
               'deer', 'dog', 
               'frog', 'horse', 
               'ship', 'truck']

# Load the CIFAR10 dataset and select only the classes 'cat' and 'dog' and 'horse'
cifar10 = cifar10_load(None)
data_path = "Dataset/data/cifar10"
(train_images, train_labels),(test_images,test_labels),y_train_df = data_preproc(data_path)

# Splt Dataset to Validation Train and Test Sets
xTrain, xVal, yTrain, yVal = train_test_split(train_images,train_labels)
xTest = test_images
yTest = test_labels

# Show dimension for each variable
print ('Train image shape:    {0}'.format(xTrain.shape))
print ('Train label shape:    {0}'.format(yTrain.shape))
print ('Validate image shape: {0}'.format(xVal.shape))
print ('Validate label shape: {0}'.format(yVal.shape))
print ('Test image shape:     {0}'.format(xTest.shape))
print ('Test label shape:     {0}'.format(yTest.shape))

class_0_index = np.where(yTrain['cat'])
print(class_0_index)
X_train_class_0 = xTrain.iloc[class_0_index]
y_train_class_0 = yTrain.iloc[class_0_index]

class_1_index = np.where(yTrain['dog'])
X_train_class_1 = xTrain.iloc[class_1_index]
y_train_class_1 = yTrain.iloc[class_1_index]

class_2_index = np.where(yTrain['horse'])
X_train_class_2 = xTrain.iloc[class_2_index]
y_train_class_2 = yTrain.iloc[class_2_index]

X = np.concatenate((X_train_class_0, X_train_class_1, X_train_class_2))
y = np.concatenate((y_train_class_0, y_train_class_1, y_train_class_2)).reshape(-1,1)

# Reshaping Dataframe to a ndarray in order to manipulate it with scikit-learn
print(X.shape) 
print(y.shape)

xTrain = np.reshape(X, (X.shape[0], -1)) 
xVal = np.reshape(xVal, (xVal.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))

# Show some CIFAR10 images -- using argmax because labels are onehot encoded
plt.subplot(221)
imshow(np.reshape(xTrain[3],newshape=(3, 32, 32)))
plt.axis('off')
plt.title(classesName[yTrain.index[np.argmax(yTrain.iloc[3])]])
plt.subplot(222)
imshow(np.reshape(xTrain[4],newshape=(3,32, 32)))
plt.axis('off')
plt.title(classesName[yTrain.index[np.argmax(yTrain.iloc[4])]])

# plt.clf()
plt.show(block=False)
plt.pause(3)
plt.close()
# print(xTrain[0])


# Reduce the training and testing data to 6000 and 1000 respectively
# xTrain, yTrain = reduce_dataset_size(xTrain, yTrain)
# xTest, yTest = reduce_dataset_size(xTest, yTest)
# Choosing a smaller dataset to work with using another way
xTrain=xTrain[:3000,:]
# use transformation that sklearn does for one_hot_encoding labels
# in order to maintain the shape of labels that should be passed to svm
yTrain=y_train_df.iloc[:3000]
yTrain = yTrain.to_numpy().reshape(-1,)
yTest = yTest.to_numpy().reshape(-1,)
yVal = yVal.to_numpy().reshape(-1,)
print(yTrain)
print(xTrain.shape)
print(yTrain.shape)

### Create an instance of the SVM classifier
model = svm.SVC(kernel='linear', C = 1, random_state=0)

print('Training the SVM classifier...')
# train model and measure training time
start_time = time.time()
model.fit(xTrain,yTrain)
end_time = time.time()
print('Training time: {}'.format(end_time - start_time))
acc_train_svm_linear = []
acc_test_svm_linear = []
# Find the prediction and accuracy on the training set.
Yhat_svc_linear_train = model.predict(xTrain)
acc_train = np.mean(Yhat_svc_linear_train == yTrain)
acc_train_svm_linear.append(acc_train)
print('Train Accuracy = {0:f}'.format(acc_train))

# Find the prediction and accuracy on the test set.
Yhat_svc_linear_test = model.predict(xVal)
print(Yhat_svc_linear_test.shape)
acc_test = np.mean(Yhat_svc_linear_test == np.argmax(yVal))
acc_test_svm_linear.append(acc_test)
print('Test Accuracy = {0:f}'.format(acc_test)) 

# Evaluate the model on the test data
accuracy = model.score(xTest, yTest.ravel())
print("Test accuracy: {:.2f}".format(accuracy))
# Print cmap

# make predictions with your classifier
y_pred = model.predict(X)         

print("Create the confusion matrix")

# optional: get true negative (tn), false positive (fp)
# false negative (fn) and true positive (tp) from confusion matrix
M = confusion_matrix(y, y_pred)
tn, fp, fn, tp = M.ravel() 
print("True-Negative:",tn)
print("False-Positive:",fp)
print("False-Negative:",fn)
print("True-Positive:",tp)
# plotting the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=M, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.show()

print('Training mutltiple SVM classifiers with varying C...')
# Create a plot for varying C parameters of the model
c_svm_linear = [0.0001,0.001,0.01,0.1,1,10,100]


for c in c_svm_linear:
    start_time = time.time()
    model = svm.SVC(probability = False,kernel='linear', C = C, random_state=0)
    end_time = time.time()
    print('Training time: {}'.format(end_time - start_time))

plt.plot(c_svm_linear, acc_train_svm_linear,'.-',color='red')
plt.plot(c_svm_linear, acc_test_svm_linear,'.-',color='orange')
plt.xlabel('c')
plt.ylabel('Accuracy')
plt.title("Plot of accuracy vs c for training and test data")
plt.grid()
plt.savefig(baseDir+'svm0.png')
plt.show()