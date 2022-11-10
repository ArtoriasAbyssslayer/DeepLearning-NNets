import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 
import glob as gb
import cv2

def load_Intel():
    DataPath = './Dataset/Intel-Image-Classification/'
    """
        Assert that the folder contains the right data
    """
    for folder in  os.listdir(DataPath + 'seg_train') : 
        files = gb.glob(pathname= str( DataPath +'seg_train/' + folder + '/*.jpg'))
        print(f'For training data , found {len(files)} in folder {folder}')
    
    for folder in  os.listdir(DataPath + 'seg_test') : 
        files = gb.glob(pathname= str( DataPath +'seg_test/' + folder + '/*.jpg'))
        print(f'For training data , found {len(files)} in folder {folder}')
    for folder in  os.listdir(DataPath + 'seg_pred') : 
        files = gb.glob(pathname= str( DataPath +'seg_pred/' + '/*.jpg'))
        print(f'For training data , found {len(files)} in folder {folder}')
    
    X_train = []
    Y_train = []
    size = 150
    classes= {'buildings':0 ,
           'forest':1,
           'glacier':2,
           'mountain':3,
           'sea':4,
           'street':5}
    for folder in os.listdir(DataPath + 'seg_train'):
        files = gb.glob(pathname=str(DataPath+'seg_train//' + folder + '/*.jpg'))
        for file in files:
            image = cv2.imread(file)
            image_array = cv2.resize(image,(size,size))
            X_train.append(list(image_array))
            Y_train.append(classes[folder])
    X_test = []
    Y_test = []
    for folder in os.listdir(DataPath + 'seg_train'):
        files = gb.glob(pathname=str(DataPath+'seg_train//' + folder + '/*.jpg'))
        for file in files:
            image = cv2.imread(file)
            image_array = cv2.resize(image,(size,size))
            X_test.append(list(image_array))
            Y_test.append(classes[folder])
    X_check = []
    Y_check = []
    for folder in os.listdir(DataPath + 'seg_train'):
        files = gb.glob(pathname=str(DataPath+'seg_train//' + folder + '/*.jpg'))
        for file in files:
            image = cv2.imread(file)
            image_array = cv2.resize(image,(size,size))
            X_check.append(list(image_array))
            Y_check.append(classes[folder])
    return X_train,Y_train,X_test,Y_test,X_check,Y_check

    