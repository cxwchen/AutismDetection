#%%
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from classification.src import classifiers as cl, basicfeatureextraction
from featureselection.src.feature_selection_methods import *
from featuredesign.graph_inference.AAL_test import *
import os,json

from nilearn.datasets import fetch_abide_pcp
from dotenv import load_dotenv
from pathlib import Path

def process_feats(
    post_process=True,
    output_dir=None,
    filt=True,
    gsr=False,
    pipeline='cpac',
    derivative='rois_aal',
    inf_method='mutual_info',
    cov_method=None,
    alpha=0.5,
    thresh=0.3,
    n_components=20,
    feats='graph',
    site=None,
    sex=None  # 'female' or 'male'
):

    # Load environment and set paths
    load_dotenv()
    abidedir = os.getenv('ABIDE_DIR_PATH')

    # Convert output_dir to Path
    if output_dir is None:
        output_dir = Path.home() / 'Documents' / 'abide_multisite_selection'
    else:
        output_dir = Path(output_dir).expanduser()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ABIDE data
    data = fetch_abide_pcp(
        derivatives=derivative,
        data_dir=abidedir,
        pipeline=pipeline,
        band_pass_filtering=filt,
        global_signal_regression=gsr,
        quality_checked=True
    )

    # Convert sex input to ABIDE encoding
    phenos = data.phenotypic
    mask = np.ones(len(phenos), dtype=bool)

    if site is not None:
        mask &= (phenos['SITE_ID'] == site)

    if sex is not None:
        sex_code = {'female': 2, 'male': 1}.get(sex.lower())
        if sex_code is None:
            raise ValueError("Invalid `sex` parameter. Use 'male', 'female', or None.")
        mask &= (phenos['SEX'] == sex_code)

    filtered_indices = np.where(mask)[0]

    filtered_data = {
        'rois_aal': [data['rois_aal'][i] for i in filtered_indices],
        'phenotypic': phenos.iloc[filtered_indices]
    }

    if post_process:
        subject_ids = [str(phen.SUB_ID).zfill(7) for phen in filtered_data['phenotypic'].itertuples()]
        data_list = [ts_file for ts_file in filtered_data['rois_aal'] if isinstance(ts_file, np.ndarray)]
        if n_components!=116:
            data_list = ica_smith(data_list, n_components=n_components)  # ICA
            #_,data_list = group_ica(data_list, n_components=n_components)

        result_df = adjacency_df(
            data_list=data_list,
            subject_ids=subject_ids,
            inf_method=inf_method,
            cov_method=cov_method,
            thresh=thresh,
            alpha=alpha,
        )

        def generate_filename(params, processing_info):
            filename_parts = [
                params['pipeline'],
                params['derivatives'].replace('_', '-'),
                'gsr' if params['global_signal_regression'] else 'nogsr',
                'filt' if params['band_pass_filtering'] else 'nofilt',
                processing_info['inf_method'].replace('_', '-'),
                processing_info['cov_method'],
                f"{processing_info['ica_components']}ICA",
                processing_info['feats'],
                f"thr{processing_info['threshold']}"
            ]
            return '_'.join(map(str, filename_parts)) + '.csv'

        processing_info = {
            'inf_method': inf_method,
            'cov_method': cov_method,
            'ica_components': n_components,
            'threshold': thresh,
            'feats': feats
        }
        params = {
            'pipeline': pipeline,
            'derivatives': derivative,
            'global_signal_regression': gsr,
            'band_pass_filtering': filt
        }

        filename = generate_filename(params, processing_info)
        output_path = output_dir / filename
        result_df.to_csv(str(output_path), index=False)

        return result_df, data_list
    else:
        return None, None
    
def main():
    inf_methods = ['norm_laplacian', 'LADMM']
    cov_methods_dict = {
        'norm_laplacian': ['direct', 'glasso', 'ledoit', 'var'],
        'LADMM': ['direct', 'var']
    }    
    n_components = 20
    for inf in inf_methods:
        for cov in cov_methods_dict[inf]:        
            print(f"Running inf_method={inf}, cov_method={cov}")
            process_feats(feats='graph', inf_method=inf, cov_method=cov, n_components=n_components, site=None)

def cross_validate_model(X, y, classifier, n_splits=5):
    """
    Perform K-Fold cross-validation on the model and return the average accuracy.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores = []

    # Convert inputs to numpy arrays once at the beginning
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train the model (example using SVM)
        model = cl.applySVM(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute accuracy for each fold
        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)

    avg_acc = np.mean(acc_scores)
    return avg_acc, acc_scores  # return both the average and individual fold scores

def iterate_and_plot_alphas(sex='all', method='rlogspect', site='NYU', alpha_value=0.9, n_splits=5):
    avg_accuracies = []
    all_fold_accuracies = []  # Store accuracies from each fold for each alpha

    print(f"Running cross-validation for alpha = {alpha_value}")
    
    # Cross-validation for current alpha
    X, y, _ = load_and_process_data(sex=sex, method=method, site=site, alpha=alpha_value)
    avg_acc, fold_accuracies = cross_validate_model(X, y, classifier="SVM", n_splits=n_splits)
    
    subject_id_to_plot = '0051044'  # You can change this subject ID if needed
    plot_adjacency_matrix(_, subject_id_to_plot)
    
    avg_accuracies.append(avg_acc)
    all_fold_accuracies.append(fold_accuracies)

    # Plotting cross-validation results
    plt.figure(figsize=(10, 6))

    # Create the fold numbers (1, 2, ..., n_splits) and their corresponding accuracies
    fold_numbers = np.arange(1, n_splits + 1)  # Fold numbers: 1, 2, ..., n_splits
    fold_accuracies_flat = fold_accuracies  # Already flattened from previous code

    # Plot each fold's accuracy
    plt.scatter(fold_numbers, fold_accuracies_flat, marker='o', color='b', label='Accuracy per fold')
    
    # Plot the average accuracy
    plt.axhline(y=avg_acc, color='r', linestyle='--', label=f'Average Accuracy = {avg_acc:.4f}')

    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy per Fold for Alpha = {alpha_value}')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot average accuracy vs alpha values
    plt.figure(figsize=(10, 6))
    plt.plot([alpha_value], avg_accuracies, marker='o', color='r', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Alpha Value')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy vs Alpha Value')
    plt.grid(True)
    plt.show()

def load_and_process_data(sex='all', method='pearson_corr', site='NYU', alpha=5):
    """
    Load data and process it for classification.
    """
    fmri_data, subject_ids, _, _ = load_files(sex=sex, max_files=800, site=site, shuffle=True, var_filt=True, ica=True)

    full_df = adjacency_df(fmri_data, subject_ids, method=method, alpha=alpha)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the DataFrame

    X = full_df.drop(columns=['DX_GROUP', 'subject_id', 'SEX'])
    y = full_df['DX_GROUP'].map({1: 1, 2: 0})  # 1 ASD, 0 ALL

    # Ensure that the data is numeric and handle missing values
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all')
    non_nan_ratio = X.notna().mean()
    X = X.loc[:, non_nan_ratio > 0.8]  # Keep columns with more than 80% non-NaN values
    X = X.loc[:, X.var() > 1e-6]  # Remove columns with low variance
    X = X.fillna(X.median())  # Fill NaN with the median

    return X, y, full_df
if __name__ == "__main__":
    main()  # Set to False if you want a single test
# %%
