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
import os,json,glob,re,random, contextlib, io
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
        output_dir = Path.home() / 'Documents' / 'abide_parameter_tuning'
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
                f"alpha{processing_info['alpha']}",
                f"thr{processing_info['threshold']}"
            ]
            return '_'.join(map(str, filename_parts)) + '.csv'

        processing_info = {
            'inf_method': inf_method,
            'cov_method': cov_method,
            'ica_components': n_components,
            'threshold': thresh,
            'alpha': alpha
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
    inf_methods = ['rlogspect']
    cov_methods_dict = {
        'rspect': ['direct'],
        'norm_laplacian': ['direct'],
        'LADMM': ['direct'],
        'rlogspect': ['direct']
    }
    alpha_values = [0.01]#np.arange(1e-1,4.6e-1,0.5e-1)
    thresholds = [0]#np.arange(0.5e-1,5e-1,5e-2)
    n_components = 20
    
    results = []
    
    for inf in inf_methods:
        for cov in cov_methods_dict[inf]:        
            for alpha in alpha_values:
                for thresh in thresholds:
                    print(f"Running inf_method={inf}, cov_method={cov}, alpha={alpha}, thresh={thresh}")
                    X, y, _ = load_and_process_data(site=None, inf_method=inf, cov_method=cov, alpha=alpha, thresh=thresh)
                    # Plot the adjacency matrix for subject '0051044' in the current subplot
                    subject_id_to_plot = '0051044'  # You can change this subject ID if needed
                    plot_adjacency_matrix(_, subject_id_to_plot)
                    avg_acc, acc_scores = cross_validate_model(X, y)
                    results.append((inf, cov, alpha, thresh, avg_acc))


    # Convert results into a numpy array for easy manipulation
    results_array = np.array(results, dtype=object)

    # Set up the grid for plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a grid based on alpha and threshold values
    grid = np.zeros((len(alpha_values), len(thresholds)))

    for i, alpha in enumerate(alpha_values):
        for j, thresh in enumerate(thresholds):
            # Extract the result corresponding to each (alpha, threshold) combination
            matching_results = results_array[(results_array[:, 2] == alpha) & (results_array[:, 3] == thresh)]
            if matching_results.size > 0:
                grid[i, j] = matching_results[0, 4]  # Store average accuracy

    # Plot the grid
    cax = ax.matshow(grid, cmap='viridis')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([f"{thresh:.2f}" for thresh in thresholds])
    ax.set_yticks(np.arange(len(alpha_values)))
    ax.set_yticklabels([f"{alpha:.1f}" for alpha in alpha_values])
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Alpha')

    # Title and display
    ax.set_title("Average Accuracy for Varying Alpha and Thresholds")
    plt.show()
            
def cross_validate_model(X, y, n_splits=5):
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

def load_and_process_data(site='NYU', inf_method='pearson_corr', alpha=5, cov_method = 'direct', thresh=0.10):
    """
    Load data and process it for classification.
    """
    full_df, _ = process_feats(inf_method=inf_method, cov_method=cov_method, alpha=alpha, thresh=thresh, site=site)
    full_df = full_df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame

    X = full_df.drop(columns=['DX_GROUP', 'subject_id', 'SEX', 'AGE_AT_SCAN'])
    y = full_df['DX_GROUP'].map({1: 1, 2: 0})  # 1 ASD, 0 ALL

    # Ensure that the data is numeric and handle missing values
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all')
    non_nan_ratio = X.notna().mean()
    X = X.loc[:, non_nan_ratio > 0.8]  # Keep columns with more than 80% non-NaN values
    X = X.loc[:, X.var() > 1e-6]  # Remove columns with low variance
    X = X.fillna(X.median())  # Fill NaN with the median

    return X, y, full_df

def plot_adjacency_matrix(df, subject_id, matrix_size=20):
    """
    This function takes the dataframe containing the flattened adjacency matrices,
    extracts the matrix for a specific subject, reshapes it, and plots the adjacency matrix.

    Parameters:
    - df: DataFrame containing the flattened adjacency matrices.
    - subject_id: The subject ID for which to plot the adjacency matrix.
    - matrix_size: The size of the square adjacency matrix (default is 20x20).
    """
    # Extract the row for the specific subject_id
    subject_row = df[df['subject_id'] == subject_id]

    # If the subject is not found in the DataFrame
    if subject_row.empty:
        print(f"Subject {subject_id} not found in the DataFrame.")
        return
    
    # Extract the flattened adjacency matrix values
    adj_values = subject_row.drop(columns=['subject_id', 'DX_GROUP', 'SEX', 'SITE_ID', 'AGE_AT_SCAN']).values.flatten()
    
    # Reshape the flattened array back into a square matrix
    adj_matrix = adj_values.reshape(matrix_size, matrix_size)
    
    # Plot the adjacency matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj_matrix, cmap="YlGnBu", annot=False, xticklabels=False, yticklabels=False)
    plt.title(f"Adjacency Matrix for Subject {subject_id}")
    plt.show()

if __name__ == "__main__":
    main()  # Set to False if you want a single test
# %%
