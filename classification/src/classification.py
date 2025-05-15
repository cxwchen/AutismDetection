import numpy as np
import pandas as pd

from classifiers import *
from hyperparametertuning import *
from loaddata import *



def performClassification(func, feat_train, feat_test, ytrain, ytest):
    model = func(feat_train, ytrain)
    ypred = model.predict(feat_test)
    
