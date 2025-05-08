import pandas as pd

def load_features(file_path):
    features = pd.read_csv(file_path)
    return features

