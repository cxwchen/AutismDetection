import os
import sys
import importlib
import datetime
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from loaddata import *
from classification import *
from classifiers import *
from hyperparametertuning import *
from performance import *

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
fs_src = os.path.join(project_root, 'featureselection', 'src')

if fs_src not in sys.path:
    sys.path.append(fs_src)

fs = importlib.import_module('feature_selection_methods')

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Create a timestamped log file
os.makedirs('logs', exist_ok=True)
log_filename = f'logs/run_{timestamp}.log'
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
comb_df = pd.concat([female_df, male_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

def runCV(df, label="female", groupeval=True, useFS=False, useHarmo=False, numfeats=100):
    # version for Jochem
    # X = df.iloc[:, 4:]
    # y = df['DX_GROUP']
    # meta = df[['SITE_ID', 'SEX', 'AGE']]

    # version for us
    X = df.iloc[:, 5:]
    y = df['DX_GROUP']
    meta = df[['SITE_ID', 'SEX', 'AGE']]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Running Stratified KFold Cross-Validation...")

    all_ytrue = {}
    all_yprob = {}
    all_ypred = {}
    for fold, (trainidx, testidx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {fold} | {label.upper()} Data ===")
        Xtrain, Xtest = X.iloc[trainidx], X.iloc[testidx]
        ytrain, ytest = y.iloc[trainidx], y.iloc[testidx]
        ytrain = ytrain.reset_index(drop=True)
        ytest = ytest.reset_index(drop=True)
        meta_train = meta.iloc[trainidx].reset_index(drop=True)
        meta_test = meta.iloc[testidx].reset_index(drop=True)

        imputer = SimpleImputer(strategy='mean')
        Xtrain = imputer.fit_transform(Xtrain)
        Xtest = imputer.transform(Xtest)
        
        # --- Harmonization with NeuroHarmonize (Optional) ---
        if useHarmo:
            print("Using NeuroHarmonize...")
            # site_train = meta['SITE_ID'].iloc[trainidx].reset_index(drop=True)
            # site_test = meta['SITE_ID'].iloc[testidx].reset_index(drop=True)

            # # NeuroHarmonize relies on Combat which requires sites in test to be seen in train
            # # Drop test sites absent in train
            # seen_sites = set(site_train)
            # mask = site_test.isin(seen_sites)
            # if (~mask).any(): #if any test site is unseen
            #     print(f"There are unseen sites. Dropping: {site_test[~mask].unique().tolist()}")

            # Xtest = Xtest[mask]
            # ytest = ytest[mask]
            # site_test = site_test[mask].reset_index(drop=True)
            # meta_test = meta_test[mask].reset_index(drop=True)
            # site_train = site_train.reset_index(drop=True)

            Xtrain, Xtest, ytest = applyHarmo(Xtrain, Xtest, meta_train, meta_test, ytest)

        # --- Feature Selection (Optional) --- 
        if useFS:
            print("Running HSIC Lasso feature selection...")
            selected_idx = fs.hsiclasso(Xtrain, ytrain, num_feat=numfeats)
            Xtrain = Xtrain[:, selected_idx]
            Xtest = Xtest[:, selected_idx]
        

        Xtrain, Xtest = normalizer(Xtrain, Xtest)

        for cfunc in [applyLogR, applySVM, applyRandForest, applyDT, applyMLP, applyLDA, applyKNN]:
            clfname = cfunc.__name__.replace("apply", "")
            print(f"\n=== Fold {fold} | {clfname}")

            if cfunc == applySVM:
                params = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, SVC())
            elif cfunc == applyDT:
                params = bestDT(Xtrain, Xtest, ytrain, ytest, DecisionTreeClassifier())
            elif cfunc == applyMLP:
                params = bestMLP(Xtrain, Xtest, ytrain, ytest, MLPClassifier())
        # rfparams = bestRF(Xtrain, Xtest, ytrain, ytest, RandomForestClassifier())
            else:
                params = None
            
            ytrue, ypred, yprob = performCA(cfunc, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp, params=params)

            all_ytrue.setdefault(clfname, []).append(ytrue)
            all_ypred.setdefault(clfname, []).append(ypred)
            if yprob is not None:
                all_yprob.setdefault(clfname, []).append(yprob)

    for clf in all_ytrue:
        ytrueAll = np.concatenate(all_ytrue[clf])
        ypredAll = np.concatenate(all_ypred[clf])
        pltAggrConfMatr(ytrueAll, ypredAll, modelname=clf, tag=label, timestamp=timestamp)
        yprobAll = np.concatenate(all_yprob[clf])
        pltROCCurve(ytrueAll, yprobAll, modelname=clf, tag=label, timestamp=timestamp)


        # performCA(applyLogR, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test)
        # performCA(applySVM, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, params = svcparams)
        # performCA(applyRandForest, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, params = None)
        # performCA(applyDT, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, params = dtparams)
        # performCA(applyMLP, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, params = mlpparams)
        # performCA(applyLDA, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test)
        # performCA(applyKNN, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test)

def runLOGO(df, label="female", useFS=False, groupeval=False, numfeats=100):
    X = df.iloc[:, 5:]
    y = df['DX_GROUP']
    meta = df[['SITE_ID', 'SEX', 'AGE']]
    sites = df['SITE_ID']

    logo = LeaveOneGroupOut()
    for fold, (trainidx, testidx) in enumerate(logo.split(X, y, groups=sites)):
        testsite = df['SITE_ID'].iloc[testidx].unique()[0]
        print(f"\n=== Fold {fold} | Testing on SITE: {testsite}")

        Xtrain, Xtest = X.iloc[trainidx], X.iloc[testidx]
        ytrain, ytest = y.iloc[trainidx], y.iloc[testidx]
        meta_test = meta.iloc[testidx].reset_index(drop=True)

        imputer = SimpleImputer(strategy='mean')
        Xtrain = imputer.fit_transform(Xtrain)
        Xtest = imputer.transform(Xtest)

        # --- Feature Selection (Optional) --- 
        if useFS:
            print("Running HSIC Lasso feature selection...")
            selected_idx = fs.hsiclasso(Xtrain, ytrain, num_feat=numfeats)
            Xtrain = Xtrain[:, selected_idx]
            Xtest = Xtest[:, selected_idx]

        Xtrain, Xtest = normalizer(Xtrain, Xtest)

        svcparams = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, SVC())
        # rfparams = bestRF(Xtrain, Xtest, ytrain, ytest, RandomForestClassifier())
        dtparams = bestDT(Xtrain, Xtest, ytrain, ytest, DecisionTreeClassifier())
        mlpparams = bestMLP(Xtrain, Xtest, ytrain, ytest, MLPClassifier())

        performCA(applyLogR, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test)
        performCA(applySVM, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, params = svcparams)
        performCA(applyRandForest, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, params = None)
        performCA(applyDT, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, params = dtparams)
        performCA(applyMLP, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, params = mlpparams)
        performCA(applyLDA, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test)
        performCA(applyKNN, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test)


def run_singlesite():
    # ============ SINGLE SITE NO FEATURE SELECTION ============================
    #Run skf cross-validation with combined data, only NYU, no feature selection
    runCV(comb_df[comb_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf_combined_onlyNYU_nofs", useFS=False, useHarmo=False)

    # Run skf cross-validation with female data, only NYU, no feature selection
    runCV(female_df[female_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf_female_onlyNYU_nofs", useFS=False, useHarmo=False)
    
    # Run skf cross-validation with male data, only NYU, no feature selection
    runCV(male_df[male_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf_male_onlyNYU_nofs", useFS=False, useHarmo=False)
    
    # ============ SINGLE SITE WITH FEATURE SELECTION ============================
    #Run skf cross-validation with combined data, only NYU, with feature selection
    runCV(comb_df[comb_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf_combined_onlyNYU_fs", useFS=True, useHarmo=False)

    # Run skf cross-validation with female data, only NYU, with feature selection
    runCV(female_df[female_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf_female_onlyNYU_fs", useFS=True, useHarmo=False)
    
    # Run skf cross-validation with male data, only NYU, with feature selection
    runCV(male_df[male_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf_male_onlyNYU_fs", useFS=True, useHarmo=False)

def run_multisite_comb():
    # runCV(female_df, label="female")
    # runCV(male_df, label="male")
    # comb_df = comb_df[comb_df['SITE_ID'] != 'CMU'].reset_index(drop=True)

    # ======================= STRATIFIED CV =================================================
    #Run skf cross-validation with combined data, harmonization=true, feature-selection=true
    runCV(comb_df[comb_df['SITE_ID'] != 'CMU'].reset_index(drop=True), label="skf_combined_harmo_fs", useFS=True, useHarmo=True)

    #Run skf cross-validation with combined data, no harmonization, no feature selection
    runCV(comb_df[comb_df['SITE_ID'] != 'CMU'].reset_index(drop=True), label="skf_combined_noharmo_nofs", useFS=False, useHarmo=False)

    #Run skf cross-validation with combined data, no harmo, with feature selection
    runCV(comb_df[comb_df['SITE_ID'] != 'CMU'].reset_index(drop=True), label="skf_combined_noharmo_fs", useFS=True, useHarmo=False)

    #Run skf cross-validation with combined data, with harmonization, no feature-selection
    runCV(comb_df[comb_df['SITE_ID'] != 'CMU'].reset_index(drop=True), label="skf_combined_harmo_nofs", useFS=False, useHarmo=True)

    # ====================== LOGO CV ==================================================
    #Run LOGO cross-validation with combined data, no feature selection
    runLOGO(comb_df, label="logo_combined_nofs", useFS=False)

    #Run LOGO cross-validation with combined data, with feature selection
    runLOGO(comb_df, label="logo_combined_fs", useFS=True)

def run_multisite_female():
    # No harmonisation applied due to very low number of entries for female data
    # run skf cross-validation with female data, no harmonisation, no feature selection
    runCV(female_df, label="skf_female_nofs", useFS=False)

    # run skf cross-validation with female data, no harmonisation, with feature selection
    runCV(female_df, label="skf_female_fs", useFS=True)

def run_multisite_male():
    # REMOVE CMU DUE TO LOW NUMBER OF ENTRIES
    #Run skf cross-validation with combined data, harmonization=true, feature-selection=true
    runCV(male_df[male_df['SITE_ID'] != 'CMU'].reset_index(drop=True), label="skf_male_harmo_fs", useFS=True, useHarmo=True)

    #Run skf cross-validation with combined data, no harmonization, no feature selection
    runCV(male_df[male_df['SITE_ID'] != 'CMU'].reset_index(drop=True), label="skf_male_noharmo_nofs", useFS=False, useHarmo=False)

    #Run skf cross-validation with combined data, no harmo, with feature selection
    runCV(male_df[male_df['SITE_ID'] != 'CMU'].reset_index(drop=True), label="skf_male_noharmo_fs", useFS=True, useHarmo=False)

    #Run skf cross-validation with combined data, with harmonization, no feature-selection
    runCV(male_df[male_df['SITE_ID'] != 'CMU'].reset_index(drop=True), label="skf_male_harmo_nofs", useFS=False, useHarmo=True)

if __name__ == "__main__":
    run_singlesite() # To run by Carmen
    # run_multisite_comb() # To run by Hannah-Rhys
    # run_multisite_female() # To run by Hannah-Rhys
    run_multisite_male() # To run by Carmen
    # print("Able to use feature selection package")