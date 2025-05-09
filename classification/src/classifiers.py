import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def applySVM(feat_train, feat_test, y):
    model = SVC() #here we should use the parameters we found in hyperparametertuning.py
    model.fit(feat_train, y)
    ypred = model.predict(feat_test)
    return ypred

def applyLogR(feat_train, feat_test, y):
    model = LogisticRegression()
    model.fit(feat_train, y)
    ypred = model.predict(feat_test)
    return ypred

def applyRandForrest(feat_train, feat_test,y):
    model = RandomForestClassifier()
    model.fit(feat_train, y)
    ypred = model.predict(feat_test)
    return ypred