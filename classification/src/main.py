import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from sklearn.neural_network import MLPClassifier

def load_features(file_path):
    features = pd.read_csv(file_path)
    return features

def get_sensitivity(ytrue, ypred):
    tn, fp, fn, tp = confusion_matrix(ytrue, ypred).ravel()
    sensitivity = tp / (tp+fn)
    return sensitivity

def get_specificity(ytrue, ypred): #in binary classification, the specificity is the same as the recall of the negative class
    tn, fp, fn, tp = confusion_matrix(ytrue, ypred).ravel()
    specificity = tn / (tn+fp)
    return specificity

def get_metrics(ytrue, ypred):
    confmatr = confusion_matrix(ytrue, ypred)
    class_report = classification_report(ytrue, ypred)
    
