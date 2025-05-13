import pandas as pd
from sklearn.model_selection import train_test_split

def load_features(file_path):
    features = pd.read_csv(file_path)
    return features

def performsplit(features, y):
    Xtrain, Xtest, ytrain, ytest = train_test_split()