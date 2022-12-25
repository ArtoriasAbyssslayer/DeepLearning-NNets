import sys
sys.path.append('..')

from sklearn.model_selection import KFold,GridSearchCV
from auxiliary_funcs.data_preprocessing import data_load_preproc
from auxiliary_funcs.data_aux import load_label_names
# import models
from models import Generic_SVM,Generic_SVM_Rbf_Kernel,Linear_SVM

label_names = load_label_names()
# Define the hyperparameter grid to search over
param_grid = {'C': [0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
              'gamma': ['scale', 'auto'],
              'shrinking': [True, False],
              }

# Define models array
models=[]
models.append(Generic_SVM())
models.append(Generic_SVM_Rbf_Kernel())
models.append(Linear_SVM())


# Define the data
(df_train_images, df_train_labels),(df_test_images,df_test_labels), y_train_df = data_load_preproc()
X = df_train_images.to_numpy().reshape()
# Define the KFold()
KFold(n_splits=5, shuffle=True, random_state=0)

def grid_search_runner(X, y, models, param_grid, scoring='accuracy', n_jobs=-1):
    """
    Run grid search on the models array
    """
    for model in models:
        print('\n\n\n\n\n Running model: {}'.format(model.__class__.__name__))
        grid_search_runner(X, y, models, param_grid)
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, njobs=n_jobs)
        grid_search.fit(X, y)
        print('Best parameters: {}'.format(grid_search.best_params_ ))
        print('Best score: {}'.format(grid_search.best_score_))
        print('Best model: {}'.format(grid_search.best_estimator_))
        print('\n\n\n\n\n')
        return grid_search

grid_search.best_estimator_.predict(xTest)

# grid_search_runner(X, y, models, params=param_grid, scoring='roc_auc')