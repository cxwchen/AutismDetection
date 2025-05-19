from performance import *
import numpy as np


def performCA(func, feat_train, feat_test, ytrain, ytest, **kwargs):
    """This function performs the classification, prediction and performance analysis
        

    """
    model = func(feat_train, ytrain, **kwargs)
    ypred = model.predict(feat_test)
    if hasattr(model, "predict_proba"):
        yprob = model.predict_proba(feat_test)[:, 1]
    elif hasattr(model, "decision_function"):
        yprob = model.decision_function(feat_test)
    else:
        raise AttributeError("Model does not support probability or decision function output.")

    # yprob = model.predict_proba(feat_test)

    metrics = get_metrics(ytest, ypred, yprob)

    clf_name = model.__class__.__name__
    plot_confusion_matrix(ytest, ypred, model)
    print_metrics(metrics, clf_name)

    

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
