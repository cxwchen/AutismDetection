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

def load_data(filepath):
    data_arrays, file_paths, subject_ids, metadata = AAL_test.load_files(filepath)
    df = AAL_test.multiset_feats(data_arrays[:5], file_paths, subject_ids) #currently 5 for testing purposes
    return df

def add_phenotypic_info(df):
    load_dotenv()
    pheno_path = os.getenv('ABIDE_PHENOTYPIC_PATH')
    df_labels = pd.read_csv(pheno_path)

    # Convert SUB_ID to match subject_id format
    df_labels['SUB_ID'] = df_labels['SUB_ID'].astype(str).str.zfill(7)
    df['subject_id'] = df['subject_id'].astype(str).str.zfill(7)

    # Select desired phenotypic columns
    df_pheno = df_labels[['SUB_ID', 'DX_GROUP', 'SEX']]

    # Merge and drop SUB_ID
    df_merged = df.merge(df_pheno, left_on='subject_id', right_on='SUB_ID', how='left')
    df_merged.drop(columns='SUB_ID', inplace=True)

    # Reorder phenotypic columns
    phenotype_cols = ['DX_GROUP', 'SEX']
    cols = phenotype_cols + [col for col in df_merged.columns if col not in phenotype_cols]
    df_merged = df_merged[cols]

    # Convert DX_GROUP from (1 = autism, 2 = control) to (1 = autism, 0 = control)
    df_merged['DX_GROUP'] = df_merged['DX_GROUP'].map({1: 1, 2: 0})

    return df_merged

def performsplit(features, y): #perform train test split for model evaluation
    Xtrain, Xtest, ytrain, ytest = train_test_split(features, y, test_size=0.3,shuffle=True, random_state=42)
    return Xtrain, Xtest, ytrain, ytest

def normalizer(feat_train, feat_test):
    scaler = StandardScaler().fit(feat_train)
    Xtrain = scaler.transform(feat_train)
    Xtest = scaler.transform(feat_test)
    return Xtrain, Xtest



# testing the functions, remove later
load_dotenv()
male_path = os.getenv('ABIDE_MALE_PATH')
female_path = os.getenv('ABIDE_FEMALE_PATH')

if not male_path or not female_path:
    raise EnvironmentError(
        "Please set ABIDE_MALE_PATH and ABIDE_FEMALE_PATH environment variables!"
    )

print("Male data path:", male_path)
print("Female data path:", female_path)
female_df = load_data(female_path)
female_df_merged = add_phenotypic_info(female_df)
