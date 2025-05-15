# Here I perform the classification for research (not GUI!)
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

from loaddata import *
from classification import *
from classifiers import *
from hyperparametertuning import *

import sys
import datetime

# Create a timestamped log file
os.makedirs('logs', exist_ok=True)
log_filename = f'logs/run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_file = open(log_filename, 'w')

# Redirect all prints to the log file and still see them in the terminal
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, log_file)
# sys.stderr = Tee(sys.stderr, log_file)  # Optional: Also capture errors


# load data
female_df = pd.read_csv("female_df_merged.csv")
X = female_df.iloc[:, 3:]
y = female_df.iloc[:, 0]

Xtrain, Xtest, ytrain, ytest = performsplit(X, y)
Xtrain, Xtest = normalizer(Xtrain, Xtest)

#Mean imputation since the features contain NaN values
imputer = SimpleImputer(strategy='mean')
Xtrain = imputer.fit_transform(Xtrain)
Xtest = imputer.transform(Xtest)

# perform classification
# performCA(applyLogR, Xtrain, Xtest, ytrain, ytest)

# svcdefault = SVC()
# params = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, svcdefault)
# performCA(applySVM, Xtrain, Xtest, ytrain, ytest, params=params)

# performCA(applyRandForest, Xtrain, Xtest, ytrain, ytest)

# performCA(applyDT, Xtrain, Xtest, ytrain, ytest)

# performCA(applyMLP, Xtrain, Xtest, ytrain, ytest)

# Binarize features
medians = np.median(Xtrain, axis=0)
feat_train_bin = (Xtrain > medians).astype(int)
feat_test_bin = (Xtest > medians).astype(int)

# Compute address size
input_size = feat_train_bin.shape[1]
addressSize = max(1, input_size // 64)

# Tune minScore
minScore_values = np.linspace(0, 1, 20)
best_minScore, _ = tune_minScore(feat_train_bin, feat_test_bin, ytrain, ytest,
                                addressSize=addressSize,
                                discriminatorLimit=4,
                                minScore_values=minScore_values)

# Run evaluation
performCA(applyClusWiSARD, feat_train_bin, feat_test_bin, ytrain, ytest, minScore=best_minScore)

# metrics blabla










