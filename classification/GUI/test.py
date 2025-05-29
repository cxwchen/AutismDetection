# -*- coding: utf-8 -*-
"""
Created on Thu May 15 15:50:07 2025

@author: kakis
"""
from HR_V1_0_03 import *
import HR_V1_0_03
import numpy as np
import pandas as pd
from classificationGUI import *
from basicfeatureextraction import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from performance import get_weights
from sklearn.neural_network import MLPClassifier

# load data

# X, y = make_classification(
#     n_samples=100,
#     n_features=10,
#     n_informative=5,
#     n_redundant=2,
#     n_classes=2,
#     random_state=42
# )

data = extract_fc_features()
data = data[data["subject_id"].str.startswith("NYU_")]
X = data.iloc[:, 3:]
y = data.iloc[:, 1]

# === Split data ===
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

svcdefault=SVC()
MLPdefault=MLPClassifier()

# params = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, hyperparametertuningGUI.param_grid, svcdefault)
# model = applySVM(Xtrain, ytrain, params)
params = bestMLP(Xtrain, Xtest, ytrain, ytest, MLPdefault)
model = applyMLP(Xtrain, ytrain, params)

shap_values = get_weights(model, Xtrain, Xtest).values

# Mean absolute SHAP values for each feature
mean_abs_shap = shap_values.mean#(axis=0)

# If you have feature names
feature_importance = pd.DataFrame({
    'feature': Xtest.columns,
    'importance': mean_abs_shap
})#.sort_values(by='importance', ascending=False)

# check()
# if __name__ == "__main__":
#     start()