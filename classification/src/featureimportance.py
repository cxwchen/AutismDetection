# this file is intended to get the feature importance
# of more complex models like MLP using SHAP

import numpy as np
from classifiers import *
from loaddata import performsplit, normalizer

def getimportanceK(model, featnames=None, k=20):
    """ 
    -----------------------------------------------------------------------
    This function is used to get the K most important features of the model
    -----------------------------------------------------------------------
    
    Parameters:
    -----------
    model : object
        Fitted estimator
    k : integer
        Choose how many of the most important features you want

    Returns
    -------
    featurelist : list
        A list of the most important features
    """
    if hasattr(model, 'coef_'): #coef_ ndarray of shape (1, n_features)
        importances = model.coef_.ravel()
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not expose feature importances")
    

    topidx = np.argsort(np.abs(importances))[::-1][:k]
    return [(featnames[i], importances[i]) for i in topidx]

def quicktest():
    df = pd.read_csv('nilearnfeatscomb.csv.gz')
    X = df.iloc[:, 4:]
    y = df['DX_GROUP']
    if set(y.unique()) == {1, 2}:
        y = y.map({1: 1, 2: 0})
    Xtrain, Xtest, ytrain, ytest = performsplit(X, y)
    Xtrain, Xtest = normalizer(Xtrain, Xtest)
    lrmodel = applyLogR(Xtrain, ytrain)
    ypred = lrmodel.predict(Xtest)
    featlist = getimportanceK(lrmodel, featnames=X.columns)
    print(featlist)

if __name__ == "__main__":
    # Quick test
    quicktest()
    