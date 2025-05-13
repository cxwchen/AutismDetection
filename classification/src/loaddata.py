import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#input van feature design groep 

def load_features(file_path): # pas aan, aan de feature design groep
    features = pd.read_csv(file_path)
    return features

def performsplit(features, y): #perform train test split for model evaluation
    Xtrain, Xtest, ytrain, ytest = train_test_split(features, y, test_size=0.3,shuffle=True, random_state=42)
    return Xtrain, Xtest, ytrain, ytest

def normalizer(feat_train, feat_test):
    scaler = StandardScaler().fit(feat_train)
    Xtrain = scaler.transform(feat_train)
    Xtest = scaler.transform(feat_test)
    return Xtrain, Xtest