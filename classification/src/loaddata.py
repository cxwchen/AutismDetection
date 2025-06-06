import sys
import os
import importlib
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neuroHarmonize import harmonizationLearn, harmonizationApply

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
feature_src = os.path.join(project_root, 'featuredesign', 'graph_inference')

if feature_src not in sys.path:
    sys.path.append(feature_src)

# print("sys.path =", sys.path)

AAL_test = importlib.import_module('AAL_test')

# print("Loading Jochems functions successfully!")


def load_data(filepath):
    data_arrays, file_paths, subject_ids, institute_names, metadata = AAL_test.load_files(filepath)
    df = AAL_test.multiset_feats(data_arrays, file_paths, subject_ids)
    return df

def add_phenotypic_info(df, save_as=None):
    load_dotenv()
    pheno_path = os.getenv('ABIDE_PHENOTYPIC_PATH')
    df_labels = pd.read_csv(pheno_path)

    # Convert SUB_ID to match subject_id format
    df_labels['SUB_ID'] = df_labels['SUB_ID'].astype(str).str.zfill(7)
    df['subject_id'] = df['subject_id'].astype(str).str.zfill(7)

    # Select desired phenotypic columns
    df_pheno = df_labels[['SUB_ID', 'SITE_ID', 'DX_GROUP', 'SEX']]

    # Merge and drop SUB_ID
    df_merged = df.merge(df_pheno, left_on='subject_id', right_on='SUB_ID', how='left')
    df_merged.drop(columns='SUB_ID', inplace=True)

    # Reorder phenotypic columns
    phenotype_cols = ['DX_GROUP', 'SEX', 'SITE_ID']
    cols = phenotype_cols + [col for col in df_merged.columns if col not in phenotype_cols]
    df_merged = df_merged[cols]

    # Convert DX_GROUP from (1 = autism, 2 = control) to (1 = autism, 0 = control)
    df_merged['DX_GROUP'] = df_merged['DX_GROUP'].map({1: 1, 2: 0})

    if save_as:
        df_merged.to_csv(f"{save_as}.csv.gz", index=False, compression='gzip')

    return df_merged

def performsplit(features, y): #perform train test split for model evaluation
    Xtrain, Xtest, ytrain, ytest = train_test_split(features, y, stratify=y, test_size=0.2,shuffle=True, random_state=42)
    return Xtrain, Xtest, ytrain, ytest

def normalizer(feat_train, feat_test):
    scaler = StandardScaler().fit(feat_train)
    Xtrain = scaler.transform(feat_train)
    Xtest = scaler.transform(feat_test)
    return Xtrain, Xtest

def applyHarmo(Xtrain, Xtest, meta_train, meta_test, ytest, ref_batch='NYU'):
    meta_train = meta_train.copy()
    meta_test = meta_test.copy()
    if meta_train['SEX'].dtype == object:
        meta_train['SEX'] = meta_train['SEX'].map({'F': 0, 'M': 1})
        meta_test['SEX'] = meta_test['SEX'].map({'F': 0, 'M': 1})

    meta_train = meta_train.rename(columns={'SITE_ID': 'SITE'})
    meta_test = meta_test.rename(columns={'SITE_ID': 'SITE'})

    seen_sites = set(meta_train['SITE'])
    mask = meta_test['SITE'].isin(seen_sites)
    if (~mask).any():
        droppedSites = meta_test.loc[~mask, 'SITE'].unique().tolist()
        print(f"Dropping unseen test sites: {droppedSites}")
        Xtest = Xtest[mask.values]
        meta_test = meta_test[mask].reset_index(drop=True)
        ytest = ytest[mask].reset_index(drop=True)

    covariates = ['SITE']
    if meta_train['SEX'].nunique() > 1:
        covariates.append('SEX')
    if meta_train['AGE'].nunique() > 1:
        covariates.append('AGE')

    print(f"Harmonizing with covariates: {covariates}")

    try:
        model, XtrainHarm = harmonizationLearn(Xtrain, covars=meta_train[covariates], ref_batch=ref_batch)

        # Make sure test covars match exactly in column names and order
        meta_test_covars = meta_test.reindex(columns=covariates)
        XtestHarm = harmonizationApply(Xtest, covars=meta_test_covars, model=model)

        return XtrainHarm, XtestHarm, ytest
    except Exception as e:
        print("ERROR during harmonization:", e)
        print("Skipping harmonization for this fold.")
        return Xtrain, Xtest, ytest
    # site_train_df = pd.DataFrame({'SITE': site_train})
    # site_test_df = pd.DataFrame({'SITE': site_test})

    # model, Xtrain_harm = harmonizationLearn(Xtrain, site_train_df)
    # Xtest_harm = harmonizationApply(Xtest, site_test_df, model)

    # return Xtrain_harm, Xtest_harm

# Perform the feature extraction and save in CSV so we won't have to keep reloading for research purposes
# For the GUI version it is not saved in a CSV but the dataframe is used directly.
if __name__ == "__main__":
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
    female_df_merged = add_phenotypic_info(female_df, save_as="female_df_merged")

    male_df = load_data(male_path)
    male_df_merged = add_phenotypic_info(male_df, save_as="male_df_merged")
