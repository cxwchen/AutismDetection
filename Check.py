#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from featuredesign.graph_inference.AAL_test import load_files, adjacency_df
from classification.src import classifiers as cl
from Pipeline import *

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

def main():
    # Initial call to set up the full_df for one fixed alpha value
    iterate_and_plot_alphas(sex='all', method='norm_laplacian', site='NYU', alpha_value=0.9)

if __name__ == '__main__':
    main()

# %%
