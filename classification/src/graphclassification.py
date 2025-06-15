import os
import sys
import glob
import datetime
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from classification import *
from classifiers import *
from hyperparametertuning import *
from performance import *
from loaddata import normalizer

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Create a timestamped log file
os.makedirs('logs', exist_ok=True)
log_filename = f'logs/run_{timestamp}.log'
log_file = open(log_filename, 'w', encoding='utf-8')

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

def runGCV(df, ncv, label="female", groupeval=True):
    df.rename(columns={
        'AGE_AT_SCAN': 'AGE',
        'subject_id': 'SUB_ID'
    }, inplace=True)

    # Define phenotypic columns if they exist
    pheno_cols = df.columns.intersection(["DX_GROUP", "SEX", "SITE_ID", "SUB_ID", "AGE"])
    X = df.drop(columns=pheno_cols)
    y = df['DX_GROUP']
    if set(y.unique()) == {1, 2}: #make sure true labels are mapped correctly
        y = y.map({1: 1, 2: 0})

    meta = df[df.columns.intersection(["SITE_ID", "SEX", "AGE"])]

    skf = StratifiedKFold(n_splits=ncv, shuffle=True, random_state=42)
    print(f"Running Stratified {ncv}Fold Cross-Validation...")

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
        for cfunc in [applyLogR, applySVM, applyRandForest, applyDT, applyMLP, applyLDA, applyKNN, applyDummy]:
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

def GordonSingleClassAll():
    load_dotenv()
    graphdir = os.getenv('GRAPHS_PATH_GORDON_SINGLE')

    if not graphdir:
        raise ValueError("GRAPHS_PATH_GORDON_SINGLE environment variable is not set.")
    
    resume_from = "cpac_rois-aal_nogsr_filt_norm-laplacian_ledoit_20ICA_graph_thr0.3.csv" #crashed during this one
    resume = False

    for fname in sorted(glob.glob(graphdir)):
        basename = os.path.basename(fname)

        if not resume:
            if basename == resume_from:
                resume = True
            else:
                print(f"Skipping {basename}")
                continue
        
        df = pd.read_csv(fname)
        label = basename.replace("cpac_rois-aal_nogsr_filt_", "").replace(".csv", "")
        runGCV(df, ncv=10, label=label)

def GordonMultiClassAll():
    load_dotenv()
    graphdir = os.getenv('GRAPHS_PATH_GORDON_MULTI_NEW')

    if not graphdir:
        raise ValueError("GRAPHS_PATH_GORDON_MULTI_NEW environment variable is not set.")

    resume_from = "cpac_rois-aal_nogsr_filt_rspect_direct_20ICA_alpha0.0001_thr0.10.csv"
    resume = False
    for fname in sorted(glob.glob(graphdir)):
        basename = os.path.basename(fname)
        if not resume:
            if basename == resume_from:
                resume = True
            else:
                print(f"Skipping {basename}")
                continue
        
        df = pd.read_csv(fname).sample(frac=1, random_state=42).reset_index(drop=True)
        label = basename.replace("cpac_rois-aal_nogsr_filt_", "").replace(".csv", "")
        runGCV(df, ncv=5, label=label)

def JochemClass():
    try:
        load_dotenv()
        graphdir = os.getenv('GRAPHS_PATH_JOCHEM')
        if not graphdir:
            raise ValueError("GRAPHS_PATH_JOCHEM is not set.")

        resume_from = "cpac_rois-aal_nogsr_filt_rlogspect_nvar_30-IC_graph_thr0.05.csv"
        resume = False

        # print(f"Loading {graphdir}")
        for fname in sorted(glob.glob(graphdir)):
            basename = os.path.basename(fname)
            if not resume:
                if basename == resume_from:
                    resume = True
                else:
                    print(f"Skipping {basename}")
                    continue
            df = pd.read_csv(fname).sample(frac=1, random_state=42).reset_index(drop=True)
            label = basename.replace("cpac_rois-aal_nogsr_filt_","").replace(".csv","")
            runGCV(df, ncv=5, label=label)
    except Exception as e:
        print("Crash in JochemClass()", e)
        import traceback
        traceback.print_exc()

def GLOGOCV(df, label="female", groupeval=False):
    df.rename(columns={
        'AGE_AT_SCAN': 'AGE',
        'subject_id': 'SUB_ID'
    }, inplace=True)

    # Define phenotypic columns if they exist
    pheno_cols = df.columns.intersection(["DX_GROUP", "SEX", "SITE_ID", "SUB_ID", "AGE"])
    X = df.drop(columns=pheno_cols)
    y = df['DX_GROUP']
    if set(y.unique()) == {1, 2}: #make sure true labels are mapped correctly
        y = y.map({1: 1, 2: 0})

    meta = df[df.columns.intersection(["SITE_ID", "SEX", "AGE"])]
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


if __name__ == "__main__":
    GordonSingleClassAll()
    JochemClass()
    GordonMultiClassAll()
    df = pd.read_csv("C:\\Users\\carme\\OneDrive\\Documenten\\AutismDetection\\Feature_Dataframes\\third_run\\cpac_rois-aal_nogsr_filt_rspect_direct_20ICA_alpha0.0001_thr0.10.csv")
    GLOGOCV(df, label="rspect_direct_20ICA_alpha0.0001_thr0.10")
