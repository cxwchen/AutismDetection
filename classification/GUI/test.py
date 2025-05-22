# -*- coding: utf-8 -*-
"""
Created on Thu May 15 15:50:07 2025

@author: kakis
"""
from HR_V1_0_03 import *
import HR_V1_0_03
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=100,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# === Split data ===
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

svcdefault=SVC()
params = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, hyperparametertuningGUI.param_grid, svcdefault)
model = applySVM(Xtrain, ytrain, params)

check()
if __name__ == "__main__":
    start()