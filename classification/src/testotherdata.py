import os
import sys
import datetime
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

from loaddata import *
from classification import *
from classifiers import *
from hyperparametertuning import *
load_dotenv()

fault_path = os.getenv('ELECTRICAL_FAULT_PATH')

data = pd.read_csv(fault_path)

features = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

data['Fault_Types'] = data[['G','C','B','A']].astype(str).agg(''.join, axis=1)

fault_types = {
    '0000' : 'No Fault',
    '1001' : 'LG Fault (Between A and Ground)',
    '0110' : 'LL Fault (Between B and C)',
    '1011' : 'LLG Fault (Between A, B, and Ground)',
    '0111' : 'LLL Fault (Between all three phases)',
    '1111' : 'GLLL Fault (Three phase symmetrical fault)'
}

data['Fault_Types'] = data['Fault_Types'].map(fault_types)

data["Fault_Types_Binary"] = data["Fault_Types_Binary"] = data["Fault_Types"].map(lambda x: 0 if x == "No Fault" else 1)

X = data.drop(["G", "C", "B", "A", "Fault_Types", "Fault_Types_Binary"], axis = 1) # axis = 1 for columns
y = data["Fault_Types_Binary"]

Xtrain, Xtest, ytrain, ytest = performsplit(X, y)
Xtrain, Xtest = normalizer(Xtrain, Xtest)
# smote = SMOTE(random_state=42)
# Xtrain, ytrain = smote.fit_resample(Xtrain, ytrain)

performCA(applyLogR, Xtrain, Xtest, ytrain, ytest)

svcdefault = SVC()
params = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, svcdefault)
performCA(applySVM, Xtrain, Xtest, ytrain, ytest, params=params)

performCA(applyRandForest, Xtrain, Xtest, ytrain, ytest)

performCA(applyDT, Xtrain, Xtest, ytrain, ytest)

performCA(applyMLP, Xtrain, Xtest, ytrain, ytest)