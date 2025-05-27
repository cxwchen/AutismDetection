import numpy as np
from dotenv import load_dotenv
import pandas as pd
import os
import re
import sys
from glob import glob
from itertools import combinations

def extract_fc_features(data_dir, pheno_path):
    phenos = pd.read_csv(pheno_path)
    phenos['SUB_ID'] = phenos['SUB_ID'].astype(str).str.zfill(7)

    all_features = []
    featnames = []
    triu_indices = None

    for file_path in sorted(glob(os.path.join(data_dir, '*.1D'))):
        base_name = os.path.basename(file_path).split('.')[0]
        match = re.match(r'.*_(\d{7})_rois_aal', base_name)
        if not match:
            print(f"Cannot recognize this filename format: {base_name}")
            continue

        subjectID = match.group(1)

        try:
            timeseries = np.loadtxt(file_path)
        except Exception as e:
            print(f"Could not load {file_path}: {e}")
            continue
    
        dx_row = phenos[phenos['SUB_ID'] == subjectID]
        if dx_row.empty:
            print(f"Phenotype not found for subject ID {subjectID} (from {base_name})")
            continue

        dx_label = dx_row['DX_GROUP'].values[0]
        sex = dx_row['SEX'].values[0]
        site = dx_row['SITE_ID'].values[0]
        age = dx_row['AGE_AT_SCAN'].values[0]
        label = 1 if dx_label == 1 else 0 # change 1=autism 2 = control to 1 = autism 0 = control

        fc_matrix = np.corrcoef(timeseries.T)
        fc_matrix = np.nan_to_num(fc_matrix) #fill NaN with 0

        if triu_indices is None:
            triu_indices = np.triu_indices_from(fc_matrix, k=1)
            n_rois = fc_matrix.shape[0]
            featnames = [f"ROI_{i}-ROI_{j}" for i, j in zip(*triu_indices)]

        fc_vec = fc_matrix[triu_indices]

        record = {
            'subject_id': subjectID,
            'SITE_ID': site,
            'SEX': 'M' if sex == 1 else 'F',
            'AGE': age,
            'DX_GROUP': label
        }

        for fname, fval in zip(featnames, fc_vec):
            record[fname] = fval
        
        all_features.append(record)
    
    df = pd.DataFrame(all_features)
    return df


if __name__ == "__main__":
    load_dotenv()
    male_path = os.getenv('ABIDE_MALE_PATH')
    female_path = os.getenv('ABIDE_FEMALE_PATH')
    phenopath = os.getenv('ABIDE_PHENOTYPIC_PATH')
    ourfeats_female = extract_fc_features(female_path, phenopath)
    ourfeats_female.to_csv("ourfeats_female.csv.gz", index=False, compression="gzip")
    ourfeats_male = extract_fc_features(male_path, phenopath)
    ourfeats_male.to_csv("ourfeats_male.csv.gz", index=False, compression="gzip")

