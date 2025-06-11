import os
import sys
import importlib
import datetime
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from loaddata import *
from classification import *
from classifiers import *
from hyperparametertuning import *
from performance import *
from visualise import *

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

comb_df = pd.read_csv("nilearnfeatscomb.csv.gz").sample(frac=1, random_state=42).reset_index(drop=True)
comb_df.rename(columns={"AGE_AT_SCAN": "AGE"}, inplace=True)
female_df = comb_df[comb_df['SEX'] == 2].sample(frac=1, random_state=42).reset_index(drop=True)
male_df = comb_df[comb_df['SEX'] == 1].sample(frac=1, random_state=42).reset_index(drop=True)

def runCV(df, label="female", groupeval=True, useFS=False, useHarmo=False, numfeats=100, ncv=5):
    df.rename(columns={
        'AGE_AT_SCAN': 'AGE',
        'subject_id': 'SUB_ID'
    }, inplace=True)

    # Define phenotypic columns if they exist
    pheno_cols = df.columns.intersection(["DX_GROUP", "SEX", "SITE_ID", "SUB_ID", "AGE"])
    X = df.drop(columns=pheno_cols)
    y = df['DX_GROUP']
    meta = df[df.columns.intersection(["SITE_ID", "SEX", "AGE"])]

    skf = StratifiedKFold(n_splits=ncv, shuffle=True, random_state=42)
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

        Xtrain, Xtest = normalizer(Xtrain, Xtest)

        # --- Feature Selection (Optional) --- 
        if useFS:
            print("Running HSIC Lasso feature selection...")
            selected_idx = fs.hsiclasso(Xtrain, ytrain, num_feat=numfeats)
            Xtrain = Xtrain[:, selected_idx]
            Xtest = Xtest[:, selected_idx]
        
        
        for cfunc in [applyLogR, applySVM, applyRandForest, applyDT, applyMLP, applyLDA, applyKNN, applyDummy]:
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
            
            ytrue, ypred, yprob, model = performCA(cfunc, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp, params=params)

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


def runLOGO(df, label="female", useFS=False, groupeval=False, numfeats=100):
    X = df.iloc[:, 4:]
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

        Xtrain, Xtest = normalizer(Xtrain, Xtest)

        # --- Feature Selection (Optional) --- 
        if useFS:
            print("Running HSIC Lasso feature selection...")
            selected_idx = fs.hsiclasso(Xtrain, ytrain, num_feat=numfeats)
            Xtrain = Xtrain[:, selected_idx]
            Xtest = Xtest[:, selected_idx]

        svcparams = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, SVC())
        # rfparams = bestRF(Xtrain, Xtest, ytrain, ytest, RandomForestClassifier())
        dtparams = bestDT(Xtrain, Xtest, ytrain, ytest, DecisionTreeClassifier())
        mlpparams = bestMLP(Xtrain, Xtest, ytrain, ytest, MLPClassifier())

        performCA(applyLogR, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp)
        performCA(applySVM, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp, params = svcparams)
        performCA(applyRandForest, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp, params = None)
        performCA(applyDT, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp, params = dtparams)
        performCA(applyMLP, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp, params = mlpparams)
        performCA(applyLDA, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp)
        performCA(applyKNN, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp)

def runCVvisu(df, label="female", groupeval=True, ncv=5):
    df.rename(columns={
        'AGE_AT_SCAN': 'AGE',
        'subject_id': 'SUB_ID'
    }, inplace=True)

    pheno_cols = df.columns.intersection(["DX_GROUP", "SEX", "SITE_ID", "SUB_ID", "AGE"])
    X = df.drop(columns=pheno_cols)
    y = df['DX_GROUP']
    meta = df[df.columns.intersection(["SITE_ID", "SEX", "AGE"])]

    skf = StratifiedKFold(n_splits=ncv, shuffle=True, random_state=42)
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

        Xtrain, Xtest = normalizer(Xtrain, Xtest)

        for cfunc in [applyLogR, applySVM]:
            clfname = cfunc.__name__.replace("apply", "")
            print(f"\n=== Fold {fold} | {clfname}")

            if cfunc == applySVM:
                params = params = {'kernel': 'linear', 'C': 1}
        # rfparams = bestRF(Xtrain, Xtest, ytrain, ytest, RandomForestClassifier())
            else:
                params = None
            
            ytrue, ypred, yprob, model = performCA(cfunc, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp, params=params)
            
            all_ytrue.setdefault(clfname, []).append(ytrue)
            all_ypred.setdefault(clfname, []).append(ypred)
            if yprob is not None:
                all_yprob.setdefault(clfname, []).append(yprob)

            plotConnectome(model, featnames=X.columns, k=20, fold=fold, tag=label, timestamp=timestamp, save_feats=True)
    
    for clf in all_ytrue:
        ytrueAll = np.concatenate(all_ytrue[clf])
        ypredAll = np.concatenate(all_ypred[clf])
        pltAggrConfMatr(ytrueAll, ypredAll, modelname=clf, tag=label, timestamp=timestamp)
        yprobAll = np.concatenate(all_yprob[clf])
        pltROCCurve(ytrueAll, yprobAll, modelname=clf, tag=label, timestamp=timestamp)

def run_singlesite():
    # ============ SINGLE SITE NO FEATURE SELECTION ============================
    # Run skf 5 fold cross-validation with combined data, only NYU, no feature selection
    runCV(comb_df[comb_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf5_combined_onlyNYU_nofs", useFS=False, useHarmo=False)

    # Run skf 5 fold cross-validation with female data, only NYU, no feature selection
    runCV(female_df[female_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf5_female_onlyNYU_nofs", useFS=False, useHarmo=False)
    
    # Run skf 5 fold cross-validation with male data, only NYU, no feature selection
    runCV(male_df[male_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf5_male_onlyNYU_nofs", useFS=False, useHarmo=False)

    # Run skf 10 fold cross-validation with combined data, only NYU, no feature selection, no group evaluation
    runCV(comb_df[comb_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf10_combined_onlyNYU_nofs", groupeval=False, useFS=False, useHarmo=False, ncv=10)

    # Run skf 10 fold cross-validation with male data, only NYU, no feature selection, no group evaluation
    runCV(male_df[male_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf10_male_onlyNYU_nofs", groupeval=False, useFS=False, useHarmo=False, ncv=10)
    
    # ============ SINGLE SITE WITH FEATURE SELECTION ============================
    # Run skf 5 fold cross-validation with combined data, only NYU, with feature selection
    runCV(comb_df[comb_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf5_combined_onlyNYU_fs", useFS=True, useHarmo=False)

    # Run skf 5 fold cross-validation with female data, only NYU, with feature selection
    runCV(female_df[female_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf5_female_onlyNYU_fs", useFS=True, useHarmo=False)
    
    # Run skf 5 fold cross-validation with male data, only NYU, with feature selection
    runCV(male_df[male_df['SITE_ID'] == 'NYU'].reset_index(drop=True), label="skf5_male_onlyNYU_fs", useFS=True, useHarmo=False)

def run_multisite_comb():
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
    runCV(male_df[male_df['SITE_ID'] != 'CMU'].reset_index(drop=True), label="skf5_male_harmo_fs", useFS=True, useHarmo=True)

    #Run skf cross-validation with combined data, no harmonization, no feature selection
    runCV(male_df[male_df['SITE_ID'] != 'CMU'].reset_index(drop=True), label="skf5_male_noharmo_nofs", useFS=False, useHarmo=False)

    #Run skf cross-validation with combined data, no harmo, with feature selection
    runCV(male_df[male_df['SITE_ID'] != 'CMU'].reset_index(drop=True), label="skf5_male_noharmo_fs", useFS=True, useHarmo=False)

    #Run skf cross-validation with combined data, with harmonization, no feature-selection
    runCV(male_df[male_df['SITE_ID'] != 'CMU'].reset_index(drop=True), label="skf5_male_harmo_nofs", useFS=False, useHarmo=True)

if __name__ == "__main__":
    run_singlesite() # To run by Carmen
    run_multisite_comb() # To run by Hannah-Rhys
    run_multisite_female() # To run by Hannah-Rhys
    run_multisite_male() # To run by Carmen
    runCVvisu(comb_df, label="skf5_combined_multisite", groupeval=True, ncv=5)
