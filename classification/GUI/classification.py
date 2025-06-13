from performance import *
import numpy as np
import pandas as pd


def performCA(func, feat_train, feat_test, ytrain, ytest, groupeval=False, fold=None, tag="", meta=None, timestamp="", **kwargs):
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

    groupeval : bool
        if True the function should perform the evaluations per subgroup.
    
    fold : integer
        the number of the fold, this can be passed to filenames and plot titles.
    
    tag : string
        Information on the dataset. Female, male, or combined. multisite or singlesite. fs or not. harmo or not

    meta : array-like
        contains phenotypic data for per-site, per-sex, and per-agegroup evaluation

    timestamp : string
        String to organise the results in their dedicated folders.
        
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
    yprob = model.predict_proba(feat_test)

    metrics = get_metrics(ytest, ypred, yprob)

    # if timestamp is not None:
        # os.makedirs(f"plots/{timestamp}", exist_ok=True)

    clf_name = model.__class__.__name__
    # plot_confusion_matrix(ytest, ypred, model, fold=fold, tag=tag, timestamp=timestamp)
    print_metrics(metrics, clf_name)
    # toCSV(DEFAULT_CSV_PATH, fold, clf_name, tag, "Overall", "ALL", metrics)

    # if groupeval and meta is not None:
    #     if 'SITE_ID' in meta.columns and meta['SITE_ID'].nunique() > 1: #no per site evaluation for single site
    #         perGroupEval(ytest, ypred, yprob, meta, group_col='SITE_ID', group_name='Site', fold=fold, classifier_name=clf_name, tag=tag)
    #     if 'SEX' in meta.columns and meta['SEX'].nunique() > 1: # only perform per sex evaluation if the df is M and F combined 
    #         perGroupEval(ytest, ypred, yprob, meta, group_col='SEX', group_name='Sex', fold=fold, classifier_name=clf_name, tag=tag)

    #     # Bin age into groups and evaluate
    #     meta = meta.copy()
    #     if 'AGE' in meta.columns:
    #         meta['AGE_GROUP'] = pd.cut(meta['AGE'], bins=[0, 11, 18, 30, 100], labels=["0-11", "12-18", "19-30", "30+"])
    #         perGroupEval(ytest, ypred, yprob, meta, group_col='AGE_GROUP', group_name='AgeGroup', fold=fold, classifier_name=clf_name, tag=tag)
    
    return ytest, ypred, yprob, model
