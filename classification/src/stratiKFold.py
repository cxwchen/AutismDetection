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
from sklearn.model_selection import StratifiedKFold


from loaddata import *
from classification import *
from classifiers import *
from hyperparametertuning import *

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

# female_df = pd.read_csv("female_df_merged.csv.gz").sample(frac=1, random_state=42).reset_index(drop=True) # shuffle the sites
# male_df = pd.read_csv("male_df_merged.csv.gz").sample(frac=1, random_state=42).reset_index(drop=True) # shuffle the sites

female_df = pd.read_csv("ourfeats_female.csv.gz").sample(frac=1, random_state=42).reset_index(drop=True) # shuffle the sites
male_df = pd.read_csv("ourfeats_male.csv.gz").sample(frac=1, random_state=42).reset_index(drop=True) # shuffle the sites

def runCV(df, label="female"):
    # version for Jochem
    # X = df.iloc[:, 4:]
    # y = df['DX_GROUP']
    # meta = df[['SITE_ID', 'SEX', 'AGE']]

    # version for us
    X = df.iloc[:, 5:]
    y = df['DX_GROUP']
    meta = df[['SITE_ID', 'SEX', 'AGE']]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (trainidx, testidx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {fold} | {label.upper()} Data ===")
        Xtrain, Xtest = X.iloc[trainidx], X.iloc[testidx]
        ytrain, ytest = y.iloc[trainidx], y.iloc[testidx]
        meta_test = meta.iloc[testidx].reset_index(drop=True)

        imputer = SimpleImputer(strategy='mean')
        Xtrain = imputer.fit_transform(Xtrain)
        Xtest = imputer.transform(Xtest)

        Xtrain, Xtest = normalizer(Xtrain, Xtest)

        svcparams = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, SVC())
        # rfparams = bestRF(Xtrain, Xtest, ytrain, ytest, RandomForestClassifier())
        dtparams = bestDT(Xtrain, Xtest, ytrain, ytest, DecisionTreeClassifier())
        mlpparams = bestMLP(Xtrain, Xtest, ytrain, ytest, MLPClassifier())

        performCA(applyLogR, Xtrain, Xtest, ytrain, ytest, fold=fold, tag=label, meta=meta_test)
        performCA(applySVM, Xtrain, Xtest, ytrain, ytest, params = svcparams, fold=fold, tag=label, meta=meta_test)
        performCA(applyRandForest, Xtrain, Xtest, ytrain, ytest, params = None, fold=fold, tag=label, meta=meta_test)
        performCA(applyDT, Xtrain, Xtest, ytrain, ytest, params = dtparams, fold=fold, tag=label, meta=meta_test)
        performCA(applyMLP, Xtrain, Xtest, ytrain, ytest, params = mlpparams, fold=fold, tag=label, meta=meta_test)
        performCA(applyLDA, Xtrain, Xtest, ytrain, ytest, fold=fold, tag=label, meta=meta_test)
        performCA(applyKNN, Xtrain, Xtest, ytrain, ytest, fold=fold, tag=label, meta=meta_test)

def run_all():
    runCV(female_df, label="female")
    runCV(male_df, label="male")
    comb_df = pd.concat([female_df, male_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    runCV(comb_df, label="combined")

if __name__ == "__main__":
    run_all()