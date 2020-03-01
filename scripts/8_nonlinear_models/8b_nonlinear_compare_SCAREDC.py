# This script cross-validates several models for predicting SCARED-C outcomes with the HBN data and outputs a summary of R2 and F1 metrics.
# The cross-validated classification models are random forest, SVC, logistic regression, and an ensemble.
# The cross-validated regression models are random forest, SVR, ridge, and an ensemble.

# Paul A. Bloom
# January 2020  

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


# Set N for cross-val iterations
n = 100

# Categorical outcomes -----------------------------------------------------------------------------

# Clean up data
hbn = pd.read_csv('../../cleanData/fullHBN.csv')
hbn = hbn.drop(['Identifiers', 'scaredSumChild', 'scaredBinParent', 'ksadsBin','scaredBinParent','ageCenter','cbclGISum'], 1).dropna(axis = 0)
hbn.reset_index(inplace = True, drop = True)
X = hbn.drop(['scaredBinChild'], axis = 1)

# scale
scaler = sk.preprocessing.StandardScaler().fit(X)
X_columns = X.columns
X = scaler.transform(X)
y = hbn['scaredBinChild']
hbn.head()

# Set up output dataframe for model metrics
df = pd.DataFrame({
    'forestTestF1':np.zeros([n])})

# Set up grid for model tuning
max_depths = [2,3]
max_features = [2,3]
min_samps = [10,15,20]

# Param grids for hyperparameter tuning
param_grid_forest = [{'max_features': max_features,
               'max_depth': max_depths,
                'min_samples_leaf': min_samps}]

param_grid_svm = [{'gamma': ['auto', 'scale'],
               'C': [.1,.5]}]

param_grid_log = [{'C': [.1,.5,1]}]


# Iteratively train/test split and save scores of each model
for i in range(n):    
	# Split data 75/25    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)

    # Randomly oversample to deal with classification of unbalanced classes    
    ros = RandomOverSampler()
    X_up, y_up = ros.fit_resample(X_train, y_train)
    
    # Set up classifiers
    forest_clf =  RandomForestClassifier(n_estimators = 100)
    svm_clf = SVC(kernel = 'rbf', gamma = 'auto')
    log_clf = LogisticRegression(C = .9, solver = 'lbfgs')
    voter_clf = VotingClassifier(estimators = [('svm', svm_clf), 
                                               ('forest', forest_clf),
                                               ('log', log_clf)])

    # Set up cv to tune each classifier
    forest_cv = GridSearchCV(forest_clf, param_grid_forest, cv = 3) 
    svm_cv = GridSearchCV(svm_clf, param_grid_svm, cv = 3) 
    log_cv = GridSearchCV(log_clf, param_grid_log, cv = 3) 
    

    # Fit each classifier        
    forestFit = forest_cv.fit(X_up, y_up)
    svcFit = svm_cv.fit(X_up, y_up)
    logFit = log_cv.fit(X_up, y_up)
    voterFit = voter_clf.fit(X_up, y_up)

    # Make predictions on test set
    forestPredTest = forestFit.predict(X_test)
    svmPredTest = svcFit.predict(X_test)
    logPredTest = logFit.predict(X_test)
    voterPredTest = voterFit.predict(X_test)
    
    # Store f1 scores in output df
    df.loc[i, 'forestTestF1'] = f1_score(y_true = y_test, y_pred = forestPredTest, average = 'macro')
    df.loc[i, 'svmTestF1'] = f1_score(y_true = y_test, y_pred = svmPredTest, average = 'macro')
    df.loc[i, 'logTestF1'] = f1_score(y_true = y_test, y_pred = logPredTest, average = 'macro')
    df.loc[i, 'voterTestF1'] = f1_score(y_true = y_test, y_pred = voterPredTest, average = 'macro')

df['outcome'] = 'SCARED-C'
df['type'] = 'Classification'


# Save output!
df.to_csv('../../output/nonlinearClassificationScaredC.csv')


# Continuous outcomes -----------------------------------------------------------------
# Clean up data
hbn = pd.read_csv('../../cleanData/fullHBN.csv')
hbn = hbn.drop(['Identifiers', 'scaredBinChild', 'scaredBinParent', 'ksadsBin','scaredBinParent','ageCenter','cbclGISum'], 1).dropna(axis = 0)
hbn.reset_index(inplace = True, drop = True)
X = hbn.drop(['scaredSumChild'], axis = 1)

# scale
scaler = sk.preprocessing.StandardScaler().fit(X)
X_columns = X.columns
X = scaler.transform(X)
y = hbn['scaredSumChild']
hbn.head()


# Set up output dataframe for model metrics
df = pd.DataFrame({
    'svrTestR2':np.zeros([n])})

# Set up grid for RF model tuning
max_depths = [2,3]
max_features = [2,3]
min_samps = [10,15,20]


# forest param grid
param_grid_forest = [{'max_features': max_features,
               'max_depth': max_depths,
                'min_samples_leaf': min_samps}]


# Iteratively train/test split and save scores of each model
for i in range(n):  
	# Split 75/25  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)
    
    # Set up regressors
    forest_reg =  RandomForestRegressor(n_estimators = 50)
    svr_reg = SVR(kernel = 'rbf', gamma = 'auto')
    ridge_reg = Ridge()
    voter_reg = VotingRegressor(estimators = [('svm', svr_reg), 
                                               ('forest', forest_reg),
                                               ('log', ridge_reg)])    

    # Tune forest regressor
    forest_cv = GridSearchCV(forest_reg, param_grid_forest, cv = 3) 
   

    # Fit regressors
    forestFit = forest_cv.fit(X_train, y_train)
    svrFit = svr_reg.fit(X_train, y_train)
    ridgeFit = ridge_reg.fit(X_train, y_train)
    voterFit = voter_reg.fit(X_train, y_train)

    # Predict on test set
    forestPredTest = forestFit.predict(X_test)
    svrPredTest = svrFit.predict(X_test)
    ridgePredTest = ridgeFit.predict(X_test)
    voterPredTest = voterFit.predict(X_test)

    # Generate scores for each model, save to df
    df.loc[i, 'forestTestR2'] = r2_score(y_true = y_test, y_pred = forestPredTest)
    df.loc[i, 'svrTestR2'] = r2_score(y_true = y_test, y_pred = svrPredTest)
    df.loc[i, 'ridgeTestR2'] = r2_score(y_true = y_test, y_pred = ridgePredTest)
    df.loc[i, 'voterTestR2'] = r2_score(y_true = y_test, y_pred = voterPredTest)

df['outcome'] = 'SCARED-C'
df['type'] = 'Regression'
# Save output!

df.to_csv('../../output/nonlinearRegressionScaredC.csv')



