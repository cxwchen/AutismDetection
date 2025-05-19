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
from imblearn.over_sampling import SMOTE

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
# female_df = pd.read_csv("female_df_merged.csv")
# X = female_df.iloc[:, 3:]
# y = female_df.iloc[:, 0]
data = pd.read_csv("abide_fc_aal.csv.gz", compression='gzip')
X = data.iloc[:, 4:]
Xfull = X.loc[:, X.columns.str.endswith('full')]
Xpart = X.loc[:, X.columns.str.endswith('partial')]
y = data.iloc[:, 3]

Xftrain, Xftest, ytrain, ytest = performsplit(Xfull, y)
Xptrain, Xptest, ytrain, ytest = performsplit(Xpart, y)

Xftrain, Xftest = normalizer(Xftrain, Xftest)
Xptrain, Xptest = normalizer(Xptrain, Xptest)

# #Mean imputation since the features contain NaN values
# imputer = SimpleImputer(strategy='mean')
# Xtrain = imputer.fit_transform(Xtrain)
# Xtest = imputer.transform(Xtest)
print("Using the AAL (116 regions) Atlas and nilearn extraction, train-test split of 20%")
print("=========================================")

print('Results for the full correlation features')

# perform classification
performCA(applyLogR, Xftrain, Xftest, ytrain, ytest)

# svcdefault = SVC()
# svcparams = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, svcdefault)
svcparams = {
    'kernel': 'linear',
    'C': 1
}
performCA(applySVM, Xftrain, Xftest, ytrain, ytest, params=svcparams)


# rfdefault = RandomForestClassifier()
# rfparams = bestRF(Xtrain, Xtest, ytrain, ytest, rfdefault)
# performCA(applyRandForest, Xtrain, Xtest, ytrain, ytest, params=rfparams)


dtdefault = DecisionTreeClassifier()
dtparams = bestDT(Xftrain, Xftest, ytrain, ytest, dtdefault)
performCA(applyDT, Xftrain, Xftest, ytrain, ytest, params=dtparams)


# mlpdefault = MLPClassifier()
# mlpparams = bestMLP(Xftrain, Xftest, ytrain, ytest, mlpdefault)
# performCA(applyMLP, Xftrain, Xftest, ytrain, ytest, params=mlpparams)


print('Results for partial correlation')

# perform classification
performCA(applyLogR, Xptrain, Xptest, ytrain, ytest)

# svcdefault = SVC()
# svcparams = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, svcdefault)
svcparams = {
    'kernel': 'linear',
    'C': 1
}
performCA(applySVM, Xptrain, Xptest, ytrain, ytest, params=svcparams)


# rfdefault = RandomForestClassifier()
# rfparams = bestRF(Xtrain, Xtest, ytrain, ytest, rfdefault)
# performCA(applyRandForest, Xtrain, Xtest, ytrain, ytest, params=rfparams)


dtdefault = DecisionTreeClassifier()
dtparams = bestDT(Xptrain, Xptest, ytrain, ytest, dtdefault)
performCA(applyDT, Xptrain, Xptest, ytrain, ytest, params=dtparams)


# mlpdefault = MLPClassifier()
# mlpparams = bestMLP(Xptrain, Xptest, ytrain, ytest, mlpdefault)
# performCA(applyMLP, Xptrain, Xptest, ytrain, ytest, params=mlpparams)










