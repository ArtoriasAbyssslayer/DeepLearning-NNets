import sys
sys.path.append('../')
from sklearn.datasets import cifar10
from sklearn.svm import SVC
from auxiliary_funcs.train_test_split import train_test_split
# Load the CIFAR-10 dataset
X, y = fetch_openml('cifar10', return_X_y=True)

# Split the dataset into training and test sets
 # Load the CIFAR-10 data    
(X_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data   
x_train = X_train.reshape(X_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255


# Train an SVM model on the training data
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = svm_model.score(X_test, y_test)
print('Accuracy on CIFAR-10 test set:', accuracy)


from sklearn.datasets import fetch_olivetti_faces

# Load the Olivetti dataset
X, y = fetch_olivetti_faces(return_X_y=True)

# Use the SVM model to classify the images in the Olivetti dataset
predictions = svm_model.predict(X)

# Calculate the accuracy of the model on the Olivetti dataset
accuracy = svm_model.score(X, y)
print('Accuracy on Olivetti dataset:', accuracy)
