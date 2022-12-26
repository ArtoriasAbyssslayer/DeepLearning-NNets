import gzip
import os
import time 
import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt
import itertools
def eval_sklearn(model, X, y):
    size = len(y)
    start = time.time()
    y_pred = model.predict(X)
    pred_dur =  time.time() - start
    # Find misclassified examples for each combination
    cm_examples = np.zeros((10,10,3,32,32))

    for x,y_i,y_pred_i in zip(X,y,y_pred):
        cm_examples[int(y_i),int(y_pred_i)] = np.reshape(x,(3,32,32))
    cm = confusion_matrix(y, y_pred)
    precisions = np.diag(cm) / np.sum(cm, axis=1)
    recalls = np.diag(cm) / np.sum(cm, axis=0)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    ncorr = sum([1 for y_pred, label in zip(pred, y) if y_pred == label])
    return accuracy,precisions,recalls,cm,cm_examples,pred_dur

# plotting functions influenced by stack overflow
def bar_plot(title,lables,data):
    idx = np.arange(len(lables))
    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.bar(idx,data)
    ax.set_title(title)
    ax.set_xticks(idx)
    ax.set_xticklabels(lables)

def plot_cm_examples(cm_examples):
    fig = plt.figure()
    fig.supxlabel('predicted classes')
    fig.supylabel('actual classes')
    for i in range(10):
        for j in range(10):
            ax = fig.add_subplot(10,10,i*10+j+1)
            img = cm_examples[i,j]
            if img.shape == (3,32,32):
                img = np.transpose(img,(1,2,0))
            ax.imshow(img)
            ax.set_title(str(i)+'->'+str(j))
            ax.axis('off')

def plot_cm(cm):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# For cros-validation resulats plots
def plot_cv_results_with_stdev(cv_results):
    pass
    Cs_rbf = [1, 10, 100, 1000]
    Cs_linear = [0.01, 0.1, 1, 10, 100, 1000]
    # plot test scores
    gammae_1_score = cv_results['mean_test_score'][0:4]
    gammae_1_score_std = cv_results['std_test_score'][0:4]
    gammae_2_score = cv_results['mean_test_score'][4:8]
    gammae_2_score_std = cv_results['std_test_score'][4:8]
    gammae_3_score = cv_results['mean_test_score'][8:12]
    gammae_3_score_std = cv_results['std_test_score'][8:12]
    gammae_4_score = cv_results['mean_test_score'][12:16]
    gammae_4_score_std = cv_results['std_test_score'][12:16]
    linear_score = cv_results['mean_test_score'][16:22]
    linear_score_std = cv_results['std_test_score'][16:22]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('accuracy')
    plt.errorbar(Cs_rbf, gammae_1_score, yerr=gammae_1_score_std, label='RBF, gamma=0.1')
    plt.errorbar(Cs_rbf, gammae_2_score, yerr=gammae_2_score_std, label='RBF, gamma=0.01')
    plt.errorbar(Cs_rbf, gammae_3_score, yerr=gammae_3_score_std, label='RBF, gamma=0.001')
    plt.errorbar(Cs_rbf, gammae_4_score, yerr=gammae_4_score_std, label='RBF, gamma=0.0001')
    plt.errorbar(Cs_linear, linear_score, yerr=linear_score_std, label='linear kerneal')
    plt.legend(loc='best')

    # plot fit time
    gammae_1_score = cv_results['mean_fit_time'][0:4]
    gammae_1_score_std = cv_results['std_fit_time'][0:4]
    gammae_2_score = cv_results['mean_fit_time'][4:8]
    gammae_2_score_std = cv_results['std_fit_time'][4:8]
    gammae_3_score = cv_results['mean_fit_time'][8:12]
    gammae_3_score_std = cv_results['std_fit_time'][8:12]
    gammae_4_score = cv_results['mean_fit_time'][12:16]
    gammae_4_score_std = cv_results['std_fit_time'][12:16]
    linear_score = cv_results['mean_fit_time'][16:22]
    linear_score_std = cv_results['std_fit_time'][16:22]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('train time (seconds)')
    plt.errorbar(Cs_rbf, gammae_1_score, yerr=gammae_1_score_std, label='RBF, gamma=0.1')
    plt.errorbar(Cs_rbf, gammae_2_score, yerr=gammae_2_score_std, label='RBF, gamma=0.01')
    plt.errorbar(Cs_rbf, gammae_3_score, yerr=gammae_3_score_std, label='RBF, gamma=0.001')
    plt.errorbar(Cs_rbf, gammae_4_score, yerr=gammae_4_score_std, label='RBF, gamma=0.0001')
    plt.errorbar(Cs_linear, linear_score, yerr=linear_score_std, label='linear kerneal')
    plt.legend(loc='best')

    # plot test time
    gammae_1_score = cv_results['mean_score_time'][0:4]
    gammae_1_score_std = cv_results['std_score_time'][0:4]
    gammae_2_score = cv_results['mean_score_time'][4:8]
    gammae_2_score_std = cv_results['std_score_time'][4:8]
    gammae_3_score = cv_results['mean_score_time'][8:12]
    gammae_3_score_std = cv_results['std_score_time'][8:12]
    gammae_4_score = cv_results['mean_score_time'][12:16]
    gammae_4_score_std = cv_results['std_score_time'][12:16]
    linear_score = cv_results['mean_score_time'][16:22]
    linear_score_std = cv_results['std_score_time'][16:22]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('test time (seconds)')
    plt.errorbar(Cs_rbf, gammae_1_score, yerr=gammae_1_score_std, label='RBF, gamma=0.1')
    plt.errorbar(Cs_rbf, gammae_2_score, yerr=gammae_2_score_std, label='RBF, gamma=0.01')
    plt.errorbar(Cs_rbf, gammae_3_score, yerr=gammae_3_score_std, label='RBF, gamma=0.001')
    plt.errorbar(Cs_rbf, gammae_4_score, yerr=gammae_4_score_std, label='RBF, gamma=0.0001')
    plt.errorbar(Cs_linear, linear_score, yerr=linear_score_std, label='linear kerneal')
    plt.legend(loc='best')