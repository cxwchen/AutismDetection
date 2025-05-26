"""Classifier methods

This script blabla



"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from wisardpkg import ClusWisard
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


def applySVM(feat_train, y, params=None, use_probabilities=True):
    """
    -----------------------------------------------------------------------------------------
    Applies Support Vector Machine (SVM) from sklearn on the training features
    -----------------------------------------------------------------------------------------

    Parameters
    ----------
    feat_train : array-like
        The training features
    y : array-like
        The true labels
    params : dict, optional
        The best parameters found using hyperparameter tuning
    use_probabilities : bool, default=True
        Whether to enable probability estimation (enables predict_proba)

    Returns
    -------
    model : object
        The trained SVM model
    """
    if params is None:
        params = {}

    # Override or inject probability setting
    params['probability'] = use_probabilities

    model = SVC(**params)
    model.fit(feat_train, y)
    return model


def applyLogR(feat_train, y):
    """
    -----------------------------------------------------------------------------------------
    This function applies Logistic Regression (LR) from sklearn on the training features
    -----------------------------------------------------------------------------------------

    Parameters
    ----------
    feat_train : array-like
        The training features
    y : array-like
        The true labels
    
    Returns
    -------
    model : object
        The trained LR model
    
    """

    model = LogisticRegression()
    model.fit(feat_train, y)
    # ypred = model.predict(feat_test)
    return model

def applyRandForest(feat_train, y, params=None):
    """
    -----------------------------------------------------------------------------------------
    This function applies Random Forest (RF) from sklearn on the training features
    -----------------------------------------------------------------------------------------

    Parameters
    ----------
    feat_train : array-like
        The training features
    y : array-like
        The true labels
    
    Returns
    -------
    model : object
        The trained RF model
    
    """
    if params is None:
            params = {}

    model = RandomForestClassifier(**params)
    model.fit(feat_train, y)
    # ypred = model.predict(feat_test)
    return model

def applyDT(feat_train, y, params=None):
    """
    -----------------------------------------------------------------------------------------
    This function applies Decision Tree (DT) from sklearn on the training features
    -----------------------------------------------------------------------------------------

    Parameters
    ----------
    feat_train : array-like
        The training features
    y : array-like
        The true labels
    
    Returns
    -------
    model : object
        The trained DT model
    
    """
    if params is None:
        params = {}
        
    model = DecisionTreeClassifier(**params)
    model.fit(feat_train, y)
    # ypred = model.predict(feat_test)
    return model

def applyMLP(feat_train, y, params=None):
    """
    -----------------------------------------------------------------------------------------
    This function applies Multi-Layer Perceptron (MLP) from sklearn on the training features
    -----------------------------------------------------------------------------------------

    Parameters
    ----------
    feat_train : array-like
        The training features
    y : array-like
        The true labels
    
    Returns
    -------
    model : object
        The trained MLP model
    
    """
    if params is None:
        params = {}

    model = MLPClassifier(**params) #default: one layer with 100 units
    model.fit(feat_train, y)
    return model

def applyClusWiSARD(feat_train, y, minScore):
    """
    -----------------------------------------------------------------------------------------
    This function applies ClusWiSARD on the training features. 
    -----------------------------------------------------------------------------------------

    To perform ClusWiSARD the features first need to be binarized

    Parameters
    ----------
    feat_train : array-like
        The training features
    y : array-like
        The true labels
    
    Returns
    -------
    model : object
        The trained MLP model
    
    """
    if not isinstance(feat_train, np.ndarray):
        feat_train = np.array(feat_train)

    if isinstance(y, pd.Series):
        y = y.to_numpy()

    Xtrain_list = feat_train.astype(int).tolist()

    # Convert labels to list of strings (or keep as strings if already)
    y_str = [str(label) for label in y]

    # model.train(Xtrain_list, y_str)
    
    # ClusWiSARD expects binary input
    medians = np.median(feat_train, axis=0)
    binarized_feat = (feat_train > medians).astype(int)

    input_size = binarized_feat.shape[1]
    addressSize = max(1, input_size // 64)

    model = ClusWisard(addressSize, minScore, 4, 1) #minScore will be added after hyperparameter tuning
    model.train(binarized_feat, y_str)
    return model


def applyLDA(feat_train, y):
    model = LinearDiscriminantAnalysis()
    model.fit(feat_train, y)
    return model

def applyKNN(feat_train, y):
    model = KNeighborsClassifier()
    model.fit(feat_train, y)
    return model