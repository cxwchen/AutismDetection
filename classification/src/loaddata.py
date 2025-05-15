import sys
import os
import importlib
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
feature_src = os.path.join(project_root, 'featuredesign', 'graph_inference')

if feature_src not in sys.path:
    sys.path.append(feature_src)

print("sys.path =", sys.path)

AAL_test = importlib.import_module('AAL_test')

print("Loading Jochems functions successfully!")

def load_data():
    load_dotenv()
    male_path = os.getenv('ABIDE_MALE_PATH')
    female_path = os.getenv('ABIDE_FEMALE_PATH')

    if not male_path or not female_path:
        raise EnvironmentError(
            "Please set ABIDE_MALE_PATH and ABIDE_FEMALE_PATH environment variables!"
        )

    print("Male data path:", male_path)
    print("Female data path:", female_path)

    mdata_arrays, mfile_paths, mmetadata = AAL_test.load_files(male_path)
    fdata_arrays, ffile_paths, fmetadata = AAL_test.load_files(female_path)

    # only 50 of each for testing purposes
    male_df = AAL_test.multiset_feats(mdata_arrays[:50], mfile_paths, output_dir="feature_outputs_male")
    female_df = AAL_test.multiset_feats(fdata_arrays[:50], ffile_paths, output_dir="feature_outputs_female")

    return male_df, female_df, mfile_paths, ffile_paths

def add_phenotypic_info(male_df, female_df, mfile_paths, ffile_paths):
    load_dotenv()
    pheno_path = os.getenv('ABIDE_PHENOTYPIC_PATH')

    return male_df, female_df

def performsplit(features, y): #perform train test split for model evaluation
    Xtrain, Xtest, ytrain, ytest = train_test_split(features, y, test_size=0.3,shuffle=True, random_state=42)
    return Xtrain, Xtest, ytrain, ytest

def normalizer(feat_train, feat_test):
    scaler = StandardScaler().fit(feat_train)
    Xtrain = scaler.transform(feat_train)
    Xtest = scaler.transform(feat_test)
    return Xtrain, Xtest



# testing the functions, remove later
male_df, female_df, mpaths, fpaths = load_data()
male_df, female_df = add_phenotypic_info(male_df, female_df, mpaths, fpaths)

male_df.head(10)

female_df.head(10)
