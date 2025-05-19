import numpy as np
from dotenv import load_dotenv
import pandas as pd
import os
import sys
import glob

import os
import glob
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from itertools import combinations

def extract_fc_features(male_path_env='ABIDE_MALE_PATH', female_path_env='ABIDE_FEMALE_PATH', phenotypic_csv_path='ABIDE_PHENOTYPIC_PATH'):
    load_dotenv()
    male_path = os.getenv(male_path_env)
    female_path = os.getenv(female_path_env)
    phenopath = os.getenv(phenotypic_csv_path)

    # Load phenotypic data
    phenos = pd.read_csv(phenopath)
    phenos['Subject_ID'] = phenos['SUB_ID'].astype(str).str.zfill(7)
    phenos['Key'] = phenos['SITE_ID'].str.upper() + '_' + phenos['Subject_ID']
    phenos['Key_upper'] = phenos['Key'].str.upper()  # Normalize for matching

    # Define AAL ROI labels (116 regions) - replace with actual AAL labels if available
    aal_labels = [f"ROI_{i+1}" for i in range(116)]

    # Generate ROI pairs for upper triangle (exclude diagonal)
    roi_pairs = list(combinations(aal_labels, 2))
    feature_names = [f"{r1}-{r2}" for r1, r2 in roi_pairs]

    data_records = []

    def process_files(file_list, sex_label):
        for file_path in file_list:
            filename = os.path.basename(file_path)
            if not filename.endswith('_rois_aal.1D'):
                continue

            base_name = filename.replace('_rois_aal.1D', '')
            base_name_upper = base_name.upper()  # Normalize case for matching

            try:
                time_series = np.loadtxt(file_path)
                fc_matrix = np.corrcoef(time_series.T)
                # Replace NaNs and inf with 0
                fc_matrix = np.nan_to_num(fc_matrix, nan=0.0, posinf=0.0, neginf=0.0)

                triu_indices = np.triu_indices_from(fc_matrix, k=1)
                fc_vector = fc_matrix[triu_indices]

                dx_row = phenos[phenos['Key_upper'] == base_name_upper]
                if dx_row.empty:
                    print(f"Phenotype not found for {base_name} (converted {base_name_upper}) from file {filename}")
                    continue

                dx_label = dx_row['DX_GROUP'].values[0]
                label = 1 if dx_label == 1 else 0

                record = {
                    'subject_id': base_name,
                    'label': label,
                    'sex': sex_label
                }

                for fname, fval in zip(feature_names, fc_vector):
                    record[fname] = fval

                data_records.append(record)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    male_files = glob.glob(os.path.join(male_path, '*rois_aal.1D'))
    female_files = glob.glob(os.path.join(female_path, '*rois_aal.1D'))

    process_files(male_files, 'M')
    process_files(female_files, 'F')

    df = pd.DataFrame(data_records)
    return df


if __name__ == "__main__":
    ourfeats = extract_fc_features()
    ourfeats.to_csv("ourfeats.csv.gz", index=False, compression="gzip")

