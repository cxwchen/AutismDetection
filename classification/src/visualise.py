import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import plotting

def getbestfeats(model, featNames, top_n=20):
    if hasattr(model, 'coef_'):
        importances = model.coef_.ravel()
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not expose feature importances")
    

    topidx = np.argsort(np.abs(importances))[::-1][:top_n]
    return [(featNames[i], importances[i]) for i in topidx]