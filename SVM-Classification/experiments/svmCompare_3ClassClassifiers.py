
import sys
sys.path.append('..')
from sklearn import svm
import time
import numpy as np
import matplotlib.pyplot as plt
from auxiliary_funcs.data_preprocessing import cifar10_load,data_preproc,train_test_split,reduce_dataset_size
from auxiliary_funcs.imshow_denormalized import imshow
from auxiliary_funcs.svm_eval_utils import plot_confusion_matrix,bar_plot
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
y = np.concatenate((y_train_class_0, y_train_class_1, y_train_class_2))

# Reshaping Dataframe to a ndarray in order to manipulate it with scikit-learn
print(X.shape) 
print(y.shape)
# change dataset to the new one
xTrain, xVal, yTrain, yVal = train_test_split(X,y)
xTrain = np.reshape(X, (X.shape[0], -1)) 
xVal = np.reshape(xVal, (xVal.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))

# Show some CIFAR10 images -- using argmax because labels are onehot encoded
image1 =  train_images.iloc[4].to_numpy()
image2 = train_images.iloc[6].to_numpy()
image3 = test_images.iloc[4].to_numpy()
image4 = test_images.iloc[6].to_numpy()

plt.subplot(221)
imshow(np.reshape(image1,(3,32,32)))
plt.axis('off')
plt.title(classesName[train_labels.index[np.argmax(train_labels.iloc[4])]])
plt.subplot(222)
imshow(np.reshape(image2,(3,32,32)))
plt.axis('off')
plt.title(classesName[train_labels.index[np.argmax(train_labels.iloc[6])]])
plt.subplot(223)
imshow(np.reshape(image3,(3,32,32)))
plt.axis('off')
plt.title(classesName[test_labels.index[np.argmax(test_labels.iloc[4])]])
plt.subplot(224)
imshow(np.reshape(image4,(3,32,32)))
plt.axis('off')
plt.title(classesName[test_labels.index[np.argmax(test_labels.iloc[6])]])


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
print("Train labels \n",yTrain)
yTrain = yTrain.to_numpy().ravel()
print("Train labels \n",yTrain)
# Coosing smaller dataset to validate with 
xVal=xVal[:1000,:]
yVal = yVal[:1000]
print("---MiniBatch for training Shape----")
print("Flatten train images array shape",xTrain.shape)
print("Train Labels shaped:",yTrain.shape)
print("---------Test Shape-----")
print(xVal.shape)
print(yVal.shape)

### Create an instance of the SVM classifier based on SVC multi-class classifier
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
acc_train = np.mean(np.sum(np.argmax(Yhat_svc_linear_train) == np.argmax(yTrain)))
acc_train_svm_linear.append(acc_train)
print('Train Accuracy = {:.2f}%'.format(acc_train*100))

# Find the prediction and accuracy on the test set.
Yhat_svc_linear_test = model.predict(xVal)
acc_test = np.mean(np.sum(np.argmax(Yhat_svc_linear_test) == np.argmax(yVal.reshape(-1,))))
acc_test_svm_linear.append(acc_test)
print('Test Accuracy Brute Force = {:.2f}%'.format(acc_test)) 

print(xVal.shape)
print(yVal.shape)
# Evaluate the model on the test data
accuracy = model.score(xVal, np.argmax(yVal,1))
print("Test accuracy with model.score: {:.2f}%".format(accuracy*100))
# Print cmap

# make predictions with your classifier
y_pred = model.predict(X)         

print("Create the confusion matrix for the three labels classified")

# optional: get true negative (tn), false positive (fp)
# false negative (fn) and true positive (tp) from confusion matrix
M = confusion_matrix(np.argmax(y).reshape(-1),np.argmax(y_pred).reshape(-1))
tn, fp, fn, tp = M.ravel() 
print("True-Negative:",tn)
print("False-Positive:",fp)
print("False-Negative:",fn)
print("True-Positive:",tp)
# plotting the confusion matrix
plot_confusion_matrix(M, ('incorrect','correct'))

print('Training mutltiple linearSVM classifiers with varying C... ')
print('And training multiple rbfSVM classifier with varying Gamm...')
# Create a plot for varying C parameters of the model
c_svm_linear = [0.0001,0.001,0.01,0.1,1,10,100]
gammavalues = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1]
acc_train_svm_linear=[]
acc_test_svm_linear=[]
acc_train_svm_rbf=[]
acc_test_svm_rbf=[]
linear_time_buffer=[]
linear_eval_time_buffer=[]
rbf_time_buffer=[]
rbf_eval_time_buffer=[]
for c,gamma in zip(c_svm_linear,gammavalues):
    start_time = time.time()
    model = svm.SVC(probability = False,kernel='linear', C = c, random_state=0)
    model_rbf = svm.SVC(probability = False,kernel='rbf', gamma=gamma, random_state=0)
    model.fit(xTrain,yTrain)
    print("Linear Model with C {}".format(c)+"\t Achieved test score {:.2f}%".format(model.score(xVal, np.argmax(yVal,1))*100))
    end_time = time.time()
    print('Training time: {}'.format(end_time - start_time))
    linear_time_buffer.append(end_time-start_time)
    eval_start = time.time()
    model.score(xVal, np.argmax(yVal,1))
    print('Evaluation time {}s'.format(time.time() - eval_start))
    linear_eval_time_buffer.append(time.time()-eval_start)
    start_time = time.time()
    model_rbf.fit(xTrain, yTrain)
    print("RBF Model with gamma {}".format(gamma)+"\t Achieved test score {:.2f}%".format(model_rbf.score(xVal, np.argmax(yVal,1))*100))
    end_time = time.time()
    print('Training time: {}'.format(end_time - start_time))
    rbf_time_buffer.append(end_time-start_time)
    eval_start = time.time()
    model_rbf.score(xVal, np.argmax(yVal,1))
    end_time = time.time()
    print('Evaluation time {}s'.format(end_time - eval_start))  
    rbf_eval_time_buffer.append(end_time-eval_start)
    acc_train_svm_linear.append(model.score(xTrain, yTrain))
    acc_test_svm_linear.append(model.score(xVal, np.argmax(yVal,1)))
    acc_train_svm_rbf.append(model_rbf.score(xTrain, yTrain))
    acc_test_svm_rbf.append(model_rbf.score(xVal, np.argmax(yVal,1)))

print('Plotting the linear_model accuracy vs C...')

plt.figure(1)
plt.plot(c_svm_linear, acc_train_svm_linear,'.-',color='red')
plt.plot(c_svm_linear, acc_test_svm_linear,'.-',color='orange')
plt.xlabel('c')
plt.ylabel('Accuracy')
plt.title("Plot of accuracy vs c for training and test data")
plt.grid()
plt.savefig('./results/svm_3classclassifier.png')
plt.show()
plt.figure(2)
bar_plot('Linear SVM varying C training times',c_svm_linear,linear_time_buffer)
plt.savefig('./results/svm_3classclassifier_times_train.png')
plt.show()
plt.figure(3)
bar_plot('Linear SVM varying C eval times',c_svm_linear,linear_eval_time_buffer)
plt.savefig('./results/svm_3classclassifier_times_eval.png')
plt.show()
print('Plotting the rbf_model accuracy vs Gamma...')
plt.figure(4)
plt.plot(gammavalues, acc_train_svm_rbf,'.-',color='red')
plt.plot(gammavalues, acc_test_svm_rbf,'.-',color='orange')
plt.xlabel('c')
plt.ylabel('Accuracy')
plt.title("Plot of accuracy vs c for training and test data")
plt.grid()
plt.savefig('./results/svm_3classclassifier_rbf.png')
plt.show()
plt.figure(5)
bar_plot('RBF SVM varying Gamma train times',gammavalues,rbf_eval_time_buffer)
plt.savefig('./results/svm_3classclassifier_rbf_times_train.png')
plt.show()
plt.figure(6)
bar_plot('RBF SVM varying Gamma eval times',gammavalues,rbf_time_buffer)
plt.savefig('./results/svm_3classclassifier_rbf_times_eval.png')
plt.show()



