
import time
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA 
from sklearn.pipeline import Pipeline
import numpy as np

import sys
sys.path.append('..')

from auxiliary_funcs.data_preprocessing import data_load_preproc
from auxiliary_funcs.train_test_split import train_test_split
def train_svm(X_train,Y_train,params=None):
    pca_varaiance = 0.9
    pca = PCA(n_components=pca_varaiance,svd_solver='full')
    svm = SVC(cache_size=200)
    pipe = Pipeline([
        ('reduce_dim',pca),
        ('clf',svm)
    ])
    if params is not None:
        pipe.set_params(**params)
    pipe.fit(X_train,Y_train)
    return pipe
import argparse
def main(args):
    images_train,labels_train,images_test,labels_test = data_load_preproc()
    params = {"clf_kearnel": "rbf", "clf_C": 100, "clf_gamma": 0.01}    
    start = time.time()
    clf = train_svm(images_train,labels_train,params)
    elapsed_time = time.time()-start
    joblib.dump(clf,args.model_name)
    
    
    # calculate metric scores
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select SVM model')
    parser.add_argument('--model_name','-m',argument_default='svm.joblib')
    args = parser.parse_args()
    main(args)
    