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
    feat_train : 
        The training features
    y : 
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

def applyRandForest(feat_train, y):
    """
    -----------------------------------------------------------------------------------------
    This function applies Random Forest (RF) from sklearn on the training features
    -----------------------------------------------------------------------------------------

    Parameters
    ----------
    feat_train : 
        The training features
    y : 
        The true labels
    
    Returns
    -------
    model : object
        The trained RF model
    
    """
    model = RandomForestClassifier()
    model.fit(feat_train, y)
    # ypred = model.predict(feat_test)
    return model

def applyDT(feat_train, y):
    """
    -----------------------------------------------------------------------------------------
    This function applies Decision Tree (DT) from sklearn on the training features
    -----------------------------------------------------------------------------------------

    Parameters
    ----------
    feat_train : 
        The training features
    y : 
        The true labels
    
    Returns
    -------
    model : object
        The trained DT model
    
    """
        
    model = DecisionTreeClassifier()
    model.fit(feat_train, y)
    # ypred = model.predict(feat_test)
    return model

def applyMLP(feat_train, y):
    """
    -----------------------------------------------------------------------------------------
    This function applies Multi-Layer Perceptron (MLP) from sklearn on the training features
    -----------------------------------------------------------------------------------------

    Parameters
    ----------
    feat_train : 
        The training features
    y : 
        The true labels
    
    Returns
    -------
    model : object
        The trained MLP model
    
    """

    model = MLPClassifier() #default: one layer with 100 units
    model.fit(feat_train, y)
    return model

def applyClusWiSARD(feat_train, y):
    """
    -----------------------------------------------------------------------------------------
    This function applies ClusWiSARD on the training features
    -----------------------------------------------------------------------------------------

    Parameters
    ----------
    feat_train : 
        The training features
    y : 
        The true labels
    
    Returns
    -------
    model : object
        The trained MLP model
    
    """
    input_size = len(feat_train[0]) ## feat_train should just have one sample but we don't know the format of our input yet
    addressSize = max(1, input_size // 64)
    model = ClusWisard(addressSize, minScore=..., discriminatorLimit=4) #minScore will be added after hyperparameter tuning
    model.fit(feat_train, y)
    return model

