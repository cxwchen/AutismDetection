import os
import sys
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from neuroHarmonize import harmonizationLearn, harmonizationApply

from classifiers import *
from performance import *
from classification import *
from hyperparametertuning import *
from loaddata import normalizer

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

female_df = pd.read_csv("ourfeats_female.csv.gz").sample(frac=1, random_state=42).reset_index(drop=True) # shuffle the sites
male_df = pd.read_csv("ourfeats_male.csv.gz").sample(frac=1, random_state=42).reset_index(drop=True) # shuffle the sites

def runHarmoCV(df, label="female"):
    X = df.iloc[:, 5:]
    y = df['DX_GROUP']
    meta = df[['SITE_ID', 'SEX', 'AGE']]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    le = LabelEncoder()
    site_encoded = le.fit_transform(meta['SITE_ID'])

    for fold, (trainidx, testidx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {fold} | {label.upper()} Data ===")
        Xtrain, Xtest = X.iloc[trainidx], X.iloc[testidx]
        ytrain, ytest = y.iloc[trainidx], y.iloc[testidx]
        site_train, site_test = site_encoded[trainidx], site_encoded[testidx]
        meta_test = meta.iloc[testidx].reset_index(drop=True)

        imputer = SimpleImputer(strategy='mean')
        Xtrain = imputer.fit_transform(Xtrain)
        Xtest = imputer.transform(Xtest)

        model, Xtrain_harm = harmonizationLearn(Xtrain, site_train)
        Xtest_harm = harmonizationApply(Xtest, site_test, model)
        Xtrain_harm, Xtest_harm = normalizer(Xtrain_harm, Xtest_harm)

        svcparams = bestSVM_RS(Xtrain_harm, Xtest_harm, ytrain, ytest, SVC())
        dtparams = bestDT(Xtrain_harm, Xtest_harm, ytrain, ytest, DecisionTreeClassifier())
        mlpparams = bestMLP(Xtrain_harm, Xtest_harm, ytrain, ytest, MLPClassifier())

        performCA(applyLogR, Xtrain_harm, Xtest_harm, ytrain, ytest, fold=fold, tag=label, meta=meta_test)
        performCA(applySVM, Xtrain_harm, Xtest_harm, ytrain, ytest, params = svcparams, fold=fold, tag=label, meta=meta_test)
        performCA(applyRandForest, Xtrain_harm, Xtest_harm, ytrain, ytest, params = None, fold=fold, tag=label, meta=meta_test)
        performCA(applyDT, Xtrain_harm, Xtest_harm, ytrain, ytest, params = dtparams, fold=fold, tag=label, meta=meta_test)
        performCA(applyMLP, Xtrain_harm, Xtest_harm, ytrain, ytest, params = mlpparams, fold=fold, tag=label, meta=meta_test)
        performCA(applyLDA, Xtrain_harm, Xtest_harm, ytrain, ytest, fold=fold, tag=label, meta=meta_test)
        performCA(applyKNN, Xtrain_harm, Xtest_harm, ytrain, ytest, fold=fold, tag=label, meta=meta_test)

def run_all():
    runHarmoCV(female_df, label="female")
    runHarmoCV(male_df, label="male")
    comb_df = pd.concat([female_df, male_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    runHarmoCV(comb_df, label="combined")

if __name__ == "__main__":
    run_all()