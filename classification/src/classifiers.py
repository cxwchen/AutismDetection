import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def applySVM(feat_train, feat_test, y):
    model = SVC()
    model.fit(feat_train, y)
    ypred = model.predict(feat_test)
    return ypred

def applyLogR(feat_train, feat_test, y):
    model = LogisticRegression()
    model.fit(feat_train, y)
    ypred = model.predict(feat_test)
    return ypred