from sklearn.neighbors  import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import s



'''
This is the part of the assignment that was issued for testing Knn classification
algorithm and see accuracy_scores in order to get hands on with some scientific 1vALL ala
'''



import data_loader_CIFAR

trainSet,testSet = data_loader_CIFAR.load_cifar10()


# class definition

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
'dog', 'frog', 'horse', 'ship', 'truck')
