import os
import sys
import glob
import datetime
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
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

        for cfunc in [applyDummy]:
        # for cfunc in [applyLogR, applySVM, applyRandForest, applyDT, applyMLP, applyLDA, applyKNN]:
            clfname = cfunc.__name__.replace("apply", "")
            print(f"\n=== Fold {fold} | {clfname}")

            if cfunc == applySVM:
                # params = {'kernel': 'linear', 'C': 1}
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
    graphdir = os.getenv('GRAPHS_PATH_GORDON_MULTI')

    if not graphdir:
        raise ValueError("GRAPHS_PATH_GORDON_MULTI environment variable is not set.")

    for fname in sorted(glob.glob(graphdir)):
        basename = os.path.basename(fname)
        
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
        # print("Data Loaded. Columns:", df.columns)
            label = basename.replace("cpac_rois-aal_nogsr_filt_","").replace(".csv","")
        # X = df.iloc[:, 4:]
        # X = X.loc[:, X.var() > 1e-6] 
        # print(X.columns)
            runGCV(df, ncv=5, label=label)
    except Exception as e:
        print("Crash in JochemClass()", e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # GordonSingleClassAll()
    # JochemClass()
    # GordonMultiClassAll()
    bestdf = pd.read_csv("C:\\Users\\carme\\OneDrive\\Documenten\\AutismDetection\\FeatDFs_Jochem\\cpac_rois-aal_nogsr_filt_rlogspect_window_30-IC_graph_thr0.05.csv")
    runGCV(bestdf, ncv=5, label="dummy_bestdf")
