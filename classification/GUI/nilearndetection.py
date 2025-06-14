# import os
# import sys
# import importlib
import datetime
import MRMR
# from dotenv import load_dotenv
# import numpy as np
# import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from loaddata import *
from classification import *
from classifiers import *
import feature_selection_methods as fs
from hyperparametertuning import *
from performance import *
# from visualise import *
from HR_V1_0_03 import *
import HR_V1_0_03

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# fs_src = os.path.join(project_root, 'featureselection', 'src')

# if fs_src not in sys.path:
#     sys.path.append(fs_src)

# fs = importlib.import_module('feature_selection_methods')

# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# # Create a timestamped log file
# os.makedirs('logs', exist_ok=True)
# log_filename = f'logs/run_{timestamp}.log'
# log_file = open(log_filename, 'w')

# # Redirect all prints to the log file and still see them in the terminal
# class Tee:
#     def __init__(self, *files):
#         self.files = files
#     def write(self, obj):
#         for f in self.files:
#             f.write(obj)
#             f.flush()
#     def flush(self):
#         for f in self.files:
#             f.flush()

# sys.stdout = Tee(sys.stdout, log_file)

# comb_df = pd.read_csv("nilearnfeatscomb.csv.gz").sample(frac=1, random_state=42).reset_index(drop=True)
# comb_df.rename(columns={"AGE_AT_SCAN": "AGE"}, inplace=True)
# female_df = comb_df[comb_df['SEX'] == 2].sample(frac=1, random_state=42).reset_index(drop=True)
# male_df = comb_df[comb_df['SEX'] == 1].sample(frac=1, random_state=42).reset_index(drop=True)

def runCV(context, label="female", groupeval=True, useHarmo=False, numfeats=100, ncv=5):
    
    useFS = context.features_set
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    skf = StratifiedKFold(n_splits=ncv, shuffle=True, random_state=42)
    print("Running Stratified KFold Cross-Validation...")

    # all_ytrue = {}
    # all_yprob = {}
    # all_ypred = {}
    
    for fold, (trainidx, testidx) in enumerate(skf.split(context.X, context.y), 1):
        print(f"\n=== Fold {fold} | {label.upper()} Data ===")
        
        Xtrain, Xtest = context.X.iloc[trainidx], context.X.iloc[testidx]
        ytrain, ytest = context.y.iloc[trainidx], context.y.iloc[testidx]
        context.ytrain = ytrain.reset_index(drop=True)
        context.ytest = ytest.reset_index(drop=True)
        # context.meta_train = context.meta.iloc[trainidx].reset_index(drop=True)
        # context.meta_test = context.meta.iloc[testidx].reset_index(drop=True)

        imputer = SimpleImputer(strategy='mean')
        context.Xtrain = imputer.fit_transform(Xtrain)
        context.Xtest = imputer.transform(Xtest)

    ### =========================================================
        # --- Feature Selection Methods (Optional) --- 
        
        if useFS != "None":
            classfier = None
            if context.mod == applySVM:
                classifier = "SVM"
            elif context.mod == applyLogR:
                classifier = "LogR"
            elif context.mod == applyDT:
                classifier = "DT"
            elif context.mod == applyMLP:
                classifier = "MLP"
            elif context.mod == applyLDA:
                classifier = "LDA"
            else:
                classifier = "KNN"
            if useFS == "hsiclasso":
                print("Running HSIC Lasso feature selection...")
                selected_idx = fs.hsiclasso(context.Xtrain, context.ytrain, classifier, num_feat=10)
                context.Xtrain = context.Xtrain[:, selected_idx]
                context.Xtest = context.Xtest[:, selected_idx]
            elif useFS == "lasso":
                print("Running Lars Lasso feature selection...")
                selected_idx = fs.lars_lasso(context.X_train, context.y_train, alpha=0.1)
                context.Xtrain = context.Xtrain[:, selected_idx]
                context.Xtest = context.Xtest[:, selected_idx]
            elif useFS == "mRMR":
                print("Running mRMR feature selection...")
                mRMR_selector = MRMR.mrmr(context.X_train, context.y_train)
                selected_idx = mRMR_selector[0:num_features_to_select]
                context.Xtrain = context.Xtrain[:, selected_idx]
                context.Xtest = context.Xtest[:, selected_idx]
            elif useFS == "SFS":
                print("Running SFS feature selection...")
                selected_idx = fs.backwards_SFS(context.X_train, context.y_train, classifier, 10)
                context.Xtrain = context.Xtrain[:, selected_idx]
                context.Xtest = context.Xtest[:, selected_idx]
                
    ### =========================================================
        
        for cfunc in [context.mod]:
            clfname = cfunc.__name__.replace("apply", "")
            print(f"\n=== Fold {fold} | {clfname}")

            if cfunc == applySVM:
                params = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, SVC())
            elif cfunc == applyDT:
                params = bestDT(Xtrain, Xtest, ytrain, ytest, DecisionTreeClassifier())
            elif cfunc == applyMLP:
                params = bestMLP(Xtrain, Xtest, ytrain, ytest, MLPClassifier())
            else:
                params = None
            
            ytrue, ypred, yprob, context.model = performCA(cfunc, context.Xtrain, context.Xtest, context.ytrain, context.ytest, groupeval=groupeval, fold=fold, tag=label, meta=context.meta_test, timestamp=timestamp, params=params)

            # all_ytrue.setdefault(clfname, []).append(ytrue)
            # all_ypred.setdefault(clfname, []).append(ypred)
            # if yprob is not None:
            #     all_yprob.setdefault(clfname, []).append(yprob)

    # for clf in all_ytrue:
    #     ytrueAll = np.concatenate(all_ytrue[clf])
    #     ypredAll = np.concatenate(all_ypred[clf])
    #     pltAggrConfMatr(ytrueAll, ypredAll, modelname=clf, tag=label, timestamp=timestamp)
    #     yprobAll = np.concatenate(all_yprob[clf])
    #     pltROCCurve(ytrueAll, yprobAll, modelname=clf, tag=label, timestamp=timestamp)


# def runLOGO(df, label="female", useFS=False, groupeval=False, numfeats=100):
#     X = df.iloc[:, 4:]
#     y = df['DX_GROUP']
#     meta = df[['SITE_ID', 'SEX', 'AGE']]
#     sites = df['SITE_ID']

#     logo = LeaveOneGroupOut()
#     for fold, (trainidx, testidx) in enumerate(logo.split(X, y, groups=sites)):
#         testsite = df['SITE_ID'].iloc[testidx].unique()[0]
#         print(f"\n=== Fold {fold} | Testing on SITE: {testsite}")

#         Xtrain, Xtest = X.iloc[trainidx], X.iloc[testidx]
#         ytrain, ytest = y.iloc[trainidx], y.iloc[testidx]
#         meta_test = meta.iloc[testidx].reset_index(drop=True)

#         imputer = SimpleImputer(strategy='mean')
#         Xtrain = imputer.fit_transform(Xtrain)
#         Xtest = imputer.transform(Xtest)

#         Xtrain, Xtest = normalizer(Xtrain, Xtest)

#         # --- Feature Selection (Optional) --- 
#         if useFS:
#             print("Running HSIC Lasso feature selection...")
#             selected_idx = fs.hsiclasso(Xtrain, ytrain, num_feat=numfeats)
#             Xtrain = Xtrain[:, selected_idx]
#             Xtest = Xtest[:, selected_idx]

#         svcparams = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, SVC())
#         # rfparams = bestRF(Xtrain, Xtest, ytrain, ytest, RandomForestClassifier())
#         dtparams = bestDT(Xtrain, Xtest, ytrain, ytest, DecisionTreeClassifier())
#         mlpparams = bestMLP(Xtrain, Xtest, ytrain, ytest, MLPClassifier())

#         performCA(applyLogR, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp)
#         performCA(applySVM, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp, params = svcparams)
#         performCA(applyRandForest, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp, params = None)
#         performCA(applyDT, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp, params = dtparams)
#         performCA(applyMLP, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp, params = mlpparams)
#         performCA(applyLDA, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp)
#         performCA(applyKNN, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp)

# def runCVvisu(df, label="female", groupeval=True, ncv=5):
#     df.rename(columns={
#         'AGE_AT_SCAN': 'AGE',
#         'subject_id': 'SUB_ID'
#     }, inplace=True)

#     pheno_cols = df.columns.intersection(["DX_GROUP", "SEX", "SITE_ID", "SUB_ID", "AGE"])
#     X = df.drop(columns=pheno_cols)
#     y = df['DX_GROUP']
#     meta = df[df.columns.intersection(["SITE_ID", "SEX", "AGE"])]

#     skf = StratifiedKFold(n_splits=ncv, shuffle=True, random_state=42)
#     print("Running Stratified KFold Cross-Validation...")

#     all_ytrue = {}
#     all_yprob = {}
#     all_ypred = {}
#     for fold, (trainidx, testidx) in enumerate(skf.split(X, y), 1):
#         print(f"\n=== Fold {fold} | {label.upper()} Data ===")
#         Xtrain, Xtest = X.iloc[trainidx], X.iloc[testidx]
#         ytrain, ytest = y.iloc[trainidx], y.iloc[testidx]
#         ytrain = ytrain.reset_index(drop=True)
#         ytest = ytest.reset_index(drop=True)
#         meta_train = meta.iloc[trainidx].reset_index(drop=True)
#         meta_test = meta.iloc[testidx].reset_index(drop=True)

#         imputer = SimpleImputer(strategy='mean')
#         Xtrain = imputer.fit_transform(Xtrain)
#         Xtest = imputer.transform(Xtest)

#         Xtrain, Xtest = normalizer(Xtrain, Xtest)

#         for cfunc in [applyLogR, applySVM]:
#             clfname = cfunc.__name__.replace("apply", "")
#             print(f"\n=== Fold {fold} | {clfname}")

#             if cfunc == applySVM:
#                 params = params = {'kernel': 'linear', 'C': 1}
#             else:
#                 params = None
            
#             ytrue, ypred, yprob, model = performCA(cfunc, Xtrain, Xtest, ytrain, ytest, groupeval=groupeval, fold=fold, tag=label, meta=meta_test, timestamp=timestamp, params=params)
            
#             all_ytrue.setdefault(clfname, []).append(ytrue)
#             all_ypred.setdefault(clfname, []).append(ypred)
#             if yprob is not None:
#                 all_yprob.setdefault(clfname, []).append(yprob)

#             plotConnectome(model, featnames=X.columns, k=20, fold=fold, tag=label, timestamp=timestamp, save_feats=True)
    
#     for clf in all_ytrue:
#         ytrueAll = np.concatenate(all_ytrue[clf])
#         ypredAll = np.concatenate(all_ypred[clf])
#         pltAggrConfMatr(ytrueAll, ypredAll, modelname=clf, tag=label, timestamp=timestamp)
#         yprobAll = np.concatenate(all_yprob[clf])
#         pltROCCurve(ytrueAll, yprobAll, modelname=clf, tag=label, timestamp=timestamp)

def run_singlesite(context, site="NYU", useFS=False):
    # ============ SINGLE SITE ============================
    # Run skf 5 fold cross-validation with combined data, only {site}
    if context.graph_vs_pearson == "Graph":
        df = pd.read_csv(context.GRAPH_FEATURES_PATH)
        if set(df["DX_GROUP"].unique()) == {1,2}:
            df["DX_GROUP"] = df["DX_GROUP"].map({1: 1, 2: 0})
            
    if context.graph_vs_pearson == "PearsonCorrelationMatrix":
        df, labels, maps, indices = nilearnextract()
        
    df.rename(columns={
        'AGE_AT_SCAN': 'AGE',
        'subject_id': 'SUB_ID'
    }, inplace=True)
    if context.subjects_sex_set > 0:
        df = df[df["SEX"] == context.subjects_sex_set]
        
    # Define phenotypic columns if they exist
    pheno_cols = df.columns.intersection(["DX_GROUP", "SEX", "SITE_ID", "SUB_ID", "AGE"])
    if context.subjects_age_set != "All":
        if 'AGE' in df.columns:
            df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 11, 18, 30, 100], labels=["0-11", "12-18", "19-30", "30+"])
        # if subject_functions[selected] != "All":
            df = df[df["AGE_GROUP"] == context.subjects_age_set]

        # Define phenotypic columns if they exist
        pheno_cols = df.columns.intersection(["DX_GROUP", "SEX", "SITE_ID", "SUB_ID", "AGE", "AGE_GROUP"])

    df = df[df['SITE_ID'] == site].reset_index(drop=True)
    
    context.X = df.drop(columns=pheno_cols)
    context.y = df['DX_GROUP']

    runCV(context, label=f"skf5_combined_only{site}_nofs", useFS=useFS)

    
    
    