from performance import *


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
