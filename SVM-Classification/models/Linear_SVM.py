"""
 LibLinear implementation 
 Solves mutli-class problem using one versus rest algorithm
 Implements Cross-Validation method to select the best C parameter value.
"""
import sklearn
import time
from sklearn.svm import LinearSVC
from auxiliary_funcs.data_preprocessing import data_load_preproc
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
# load dataset
Images_train, Labels_train, Images_test, Labels_test = data_load_preproc()
svm = LinearSVC(loss='hinge', multi_class='ovr',C=100, max_iter=1000)

# Define Kfolds splitting method
kfold = KFold(n_splits=5,shuffle=True,random_state=0)
class LinearSVM(object):
    def __init__(self, C=1.0, loss='hinge', multi_class='ovr', max_iter=1000):
        self.C = C
        self.loss = loss
        self.multi_class = multi_class
        self.max_iter = max_iter
        self.svm =  LinearSVC(loss=self.loss, multi_class=self.multi_class,C=self.C, max_iter=self.max_iter)
    def train(Images_train,Labels_train):
        tic = time.time()
        self.svm.fit(Images_train, Labels_train)
        toc =  time.time()
        print("Elapsed time {:.2f} seconds".format(toc - tic))
        return 0
    def cv_eval(Images_test, Labels_test):
        pred =  self.svm.predict(Images_test)
        cv_score = cross_val_score(self.svm, Images_test, Labels_test, cv=kfold, scoring='accuracy')
        print("Accuracy: {:.2f}".format(cv_score.mean()))
        return cv_score
    def cv_pred(Images_test):
        prediction = self.svm.predict(Images_test)
        return prediction
    