from performance import *
import numpy as np


def performCA(func, feat_train, feat_test, ytrain, ytest, fold=None, tag="", meta=None, **kwargs):
    """
    --------------------------------------------------------------------------------
    This function performs the classification, prediction and performance analysis
    --------------------------------------------------------------------------------

    Parameters
    ----------
    func :
        The classifier functions to use. Options: applyLogR, applySVM, applyRandForest, applyDT, applyMLP, applyLDA, and applyKNN
    
    feat_train : array-like
        the features that are used for training
    
    feat_test : array-like
        the features that are used for validation
    
    ytrain : array-like
        the true labels of the training set
    
    ytest : array-like
        the true labels of the validation set
    
    fold : integer
        the number of the fold
    
    tag : string
        Information on the dataset. Either female, male, or combined

    meta : array-like
        contains phenotypic data for per-site or per-sex evaluation
        
    **kwargs : dict
        to pass parameters found using hyperparameter tuning. Default: params = None

    """
    model = func(feat_train, ytrain, **kwargs)
    
    if hasattr(model, "predict_proba"):
        yprob = model.predict_proba(feat_test)[:, 1]
    elif hasattr(model, "decision_function"):
        yprob = model.decision_function(feat_test)
    else:
        yprob = None

    ypred = model.predict(feat_test)
    # yprob = model.predict_proba(feat_test)

    metrics = get_metrics(ytest, ypred, yprob)

    clf_name = model.__class__.__name__
    plot_confusion_matrix(ytest, ypred, model, fold=fold, tag=tag)
    print_metrics(metrics, clf_name)
    toCSV(DEFAULT_CSV_PATH, fold, clf_name, tag, "Overall", "ALL", metrics)

    if meta is not None:
        perGroupEval(ytest, ypred, yprob, meta, group_col='SITE_ID', group_name='Site', fold=fold, classifier_name=clf_name, tag=tag)
        if 'SEX' in meta.columns and meta['SEX'].nunique() > 1: # only perform per sex evaluation if the df is M and F combined 
            perGroupEval(ytest, ypred, yprob, meta, group_col='SEX', group_name='Sex', fold=fold, classifier_name=clf_name, tag=tag)

    

## separate function for ClusWiSARD
def performCA_cluswisard(func, feat_train, feat_test, ytrain, ytest, **kwargs):
    """
    Performs classification and evaluation specifically for ClusWiSARD.
    """
    model = func(feat_train, ytrain, **kwargs)

    # Predict using ClusWiSARD
    ypred = model.classify(feat_test)

    # ClusWiSARD doesn't support probabilities -> use dummy values for AUROC
    # We'll use 1 for predicted class 1 and 0 for class 0
    yprob = np.array(ypred, dtype=float)

    # Get metrics (robust to missing probability info)
    metrics = get_metrics(ytest, ypred, yprob)

    clf_name = model.__class__.__name__
    plot_confusion_matrix(ytest, ypred, model)
    print_metrics(metrics, clf_name)
