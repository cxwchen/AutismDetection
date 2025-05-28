import os
from nilearn.datasets import fetch_abide_pcp
from nilearn.connectome import ConnectivityMeasure
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

# Function to process one atlas
def extract_fc_df(derivative_name, label):
    data = fetch_abide_pcp(derivatives=derivative_name, **params)
    conn_full = ConnectivityMeasure(kind='correlation')
    conn_partial = ConnectivityMeasure(kind='partial correlation')
    phenos = data.phenotypic
    time_series_files = data[derivative_name]

    records = []

    for ts_file, phen in tqdm(zip(time_series_files, phenos.itertuples())):
        try:
            if not isinstance(ts_file, np.ndarray):
                raise ValueError("Expected time series as a NumPy array")
            time_series = ts_file

            full_corr = conn_full.fit_transform([time_series])[0]
            partial_corr = conn_partial.fit_transform([time_series])[0]

            triu_idx = np.triu_indices_from(full_corr, k=1) # exclude diagonal
            full_vec = full_corr[triu_idx]
            partial_vec = partial_corr[triu_idx]

            record = {
                'subject_id': phen.SUB_ID,
                'site': phen.SITE_ID,
                'sex': phen.SEX,
                'label': 1 if phen.DX_GROUP == 1 else 0 # change TC = 2 to TC = 0 for classfiers
            }

            if not records:
                n_rois = time_series.shape[1]
                feature_names = [f'ROI_{i}_{j}' for i, j in zip(*triu_idx)]

            for name, fval, pval in zip(feature_names, full_vec, partial_vec):
                record[f'{name}_full'] = fval
                record[f'{name}_partial'] = pval

            records.append(record)
        except Exception as e:
            print(f"[{label}] Failed for subject {phen.SUB_ID}: {e}")

    return pd.DataFrame.from_records(records)

if __name__ == "__main__":
    load_dotenv()
    abidedir = os.getenv('ABIDE_DIR_PATH')

    params = dict(data_dir=abidedir, pipeline='cpac', band_pass_filtering=True, quality_checked=True)

    atlases = {
            'AAL': 'rois_aal',
            'CC200': 'rois_cc200',
            'CC400': 'rois_cc400'
    }

    df_aal = extract_fc_df('rois_aal', label='AAL')
    df_cc200 = extract_fc_df('rois_cc200', label='CC200')
    df_cc400 = extract_fc_df('rois_cc400', label='CC400')

    df_aal.to_csv("abide_fc_aal.csv.gz", index=False, compression='gzip')
    df_cc200.to_csv("abide_fc_cc200.csv.gz", index=False, compression='gzip')
    df_cc400.to_csv("abide_fc_cc400.csv.gz", index=False, compression='gzip')

    df_aal.to_csv("abide_fc_aal.csv", index=False)
    df_cc200.to_csv("abide_fc_cc200.csv", index=False)
    df_cc400.to_csv("abide_fc_cc400.csv", index=False)