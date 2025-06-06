#%%
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from classification.src import classifiers as cl
from featureselection.src.feature_selection_methods import *
from featureselection.src import cluster
from featureselection.src import Compute_HSIC_Lasso as hsic_lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from featuredesign.graph_inference.AAL_test import multiset_feats, load_files, adjacency_df
import cvxpy as cp
import seaborn as sns

def load_file(sex='all', method='pearson_corr'):
    #folder_path = r"C:\Users\guus\Python_map\AutismDetection-main\abide\female-cpac-filtnoglobal-aal" # Enter your local ABIDE dataset path
    fmri_data, subject_ids, _, _ = load_files(sex=sex, max_files=800, shuffle=True, var_filt=True, ica=True)

    print(f"Final data: {len(fmri_data)} subjects")
    print(f"Final IDs: {len(subject_ids)}")

    full_df = adjacency_df(fmri_data, subject_ids, method = 'norm_laplacian')
    print("Merged feature+label shape:\n", full_df.shape)

    print(full_df)
    
    subject_id_to_plot = '0050524'  # Change this to any valid subject ID
    plot_adjacency_matrix(full_df, subject_id_to_plot)
    
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the DataFrame

    X = full_df.drop(columns=['DX_GROUP', 'subject_id', 'SEX'])
    y = full_df['DX_GROUP'].map({1: 1, 2: 0})

    # Making sure the data is numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1,how='all')
    non_nan_ratio = X.notna().mean()
    X = X.loc[:, non_nan_ratio > 0.8]  # Keep columns with more than 50% non-NaN values
    # Making sure there is no 0 var data for the hsic algorithm
    X = X.loc[:, X.var() > 1e-6]

    # NaN values are filled with the median of the column
    X= X.fillna(X.median())

    return X, y, fmri_data

def evaluate_performance(y_true, y_pred, y_proba=None, show_plots=False, classifier_name="", fold_idx=None, verbose=True):
    # Compute basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    if verbose==True:
        print(f"\nPerformance Metrics ({classifier_name}, Fold {fold_idx if fold_idx is not None else ''}):")
        print(f"Performance Metrics ({classifier_name}):")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        if auc is not None:
            print(f"  AUC:       {auc:.4f}")

    if show_plots:
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {classifier_name}")
        plt.show()

        # ROC curve (if proba is available)
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {classifier_name}")
            plt.legend()
            plt.grid()
            plt.show()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc
    }

def print_selected_features(selected_features, selected_feature_names, print_feat=False):
    print("Selected features:", selected_features)
    if print_feat==True:
        print(f"\nSelected feature names({len(selected_feature_names)}):")
        for name in selected_feature_names:
            print("-", name)

def train_and_evaluate(X, y, classifier):
    #splitting the data in train and test 0.8:0.2 respecively
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, stratify=y)

    #scale the data for the classifier
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    if classifier == "SVM":
        model_raw = cl.applySVM(X_train_scaled, y_train)
    elif classifier == "RandomForest":
        model_raw = cl.applyRandForest(X_train_scaled, y_train)
    elif classifier == "LogR":
        model_raw = cl.applyLogR(X_train_scaled, y_train)
    elif classifier == "DecisionTree":
        model_raw = cl.applyDT(X_train_scaled, y_train)
    elif classifier == "MLP":
        model_raw = cl.applyMLP(X_train_scaled, y_train)
    else:
        print("Classifier not supported: choose from SVM, RandomForest, LogR, DecisionTree or MLP")
    #applying the classifier to the total data
    model_raw = cl.applySVM(X_train, y_train)
    y_pred_raw = model_raw.predict(X_test)

    try:
        y_proba_raw = model_raw.predict_proba(X_test_scaled)[:, 1]
    except:
        y_proba_raw = None

    try:
        y_proba_raw = model_raw.predict_proba(X_test_scaled)[:, 1]
    except:
        y_proba_raw = None

    #finding mse and accuracy
    perf_raw = evaluate_performance(y_test, y_pred_raw, y_proba_raw, classifier_name=classifier, verbose=False)
    acc_raw = perf_raw["accuracy"]
    mse_raw = mean_squared_error(y_test, y_pred_raw)
    precision_raw = perf_raw["precision"]
    recall_raw = perf_raw["recall"]
    F1_raw = perf_raw["f1"]
    AUC_raw = perf_raw["auc"]
    print(classification_report(y_test, y_pred_raw, target_names=["Class 0", "Class 1"]))
    print('Confusion matrix:', confusion_matrix(y_test, y_pred_raw))
    print('Amount of features:', X_train.shape[1])

    #acc, mse, selected_feature_names = cross_validate_model(X, y, selected_features)
    print(f"Train/Test Accuracy raw: {acc_raw:.4f}, MSE: {mse_raw:.4f}, Precision: {precision_raw:.4f}, Recall: {recall_raw:.4f}, F1: {F1_raw:.4f}, AUC: {AUC_raw:.4f}")

    return X_train, X_test, y_train, y_test

def classify(X_train, X_test, y_train, y_test, selected_features, classifier, performance=True):

    #scale the data for the classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selected_train_x = X_train_scaled.iloc[:, selected_features]
    selected_test_x = X_test_scaled.iloc[:, selected_features]

    if classifier == "SVM":
        model = cl.applySVM(selected_train_x, y_train)
    elif classifier == "RandomForest":
        model = cl.applyRandForest(selected_train_x, y_train)
    elif classifier == "LogR":
        model = cl.applyLogR(selected_train_x, y_train)
    elif classifier == "DecisionTree":
        model = cl.applyDT(selected_train_x, y_train)
    elif classifier == "MLP":
        model = cl.applyMLP(selected_train_x, y_train)
    else:
        print("Classifier not supported: choose from SVM, RandomForest, LogR, DecisionTree or MLP")
    
    #applying the classifier to the selected data
    y_pred = model.predict(selected_test_x)
    #params=bestSVM_RS(X_train, X_test, y_train, y_test, svcdefault=SVC())
    #finding mse and accuracy

    # Predict probabilities if supported
    try:
        y_proba = model.predict(selected_test_x) if hasattr(model, "predict_proba") else None
        if y_proba is not None:
            y_proba = model.predict_proba(selected_test_x)[:, 1]
    except:
        y_proba = None
    if performance==True:
        evaluate_performance(y_test, y_pred, y_proba, classifier_name=classifier)
    #getting and printing the feature names
    feature_names = X_train.columns
    selected_feature_names = feature_names[selected_features]
    
    return selected_feature_names

def cross_validate_model(X, y, feature_selection, classifier, raw=True, n_splits=5, **feature_selection_kwargs):
    #K-Fold cross-validation evaluation.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores = []
    mse_scores = []
    precision_scores = []
    recall_scores = []
    F1_scores = []
    AUC_scores = []
    acc_scores_raw = []
    mse_scores_raw = []
    precision_scores_raw = []
    recall_scores_raw = []
    F1_scores_raw = []
    AUC_scores_raw= []

    # Convert inputs to numpy arrays once at the beginning
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        X = X.to_numpy()
    elif isinstance(X, pd.Series):
        feature_names = [f"feature_{i}" for i in range(len(X))]
        X = X.to_numpy()
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
    X = np.asarray(X, dtype=np.float64)

    # Inside failsafe_feature_selection
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = y.values
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    
    selected_features = None
    selected_feature_names = None

    for train_idx, test_idx in kf.split(X):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        #Scaling the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        selected_features = failsafe_feature_selection(feature_selection, X_train_scaled, y_train, classifier=classifier, **feature_selection_kwargs)

        # Ensure selected_features is a list of valid indices
        if not isinstance(selected_features, (list, np.ndarray)):
            selected_features = [selected_features] if selected_features is not None else []
            
        selected_features = [int(idx) for idx in selected_features if isinstance(idx, (int, np.integer)) and 0 <= idx < X_train.shape[1]]
            
        if not selected_features:
            # Fallback to all features if selection fails
            selected_features = list(range(X_train.shape[1]))

        # Select the features based on the selected indices 
        X_train_sel = X_train_scaled[:, selected_features]
        X_test_sel = X_test_scaled[:, selected_features]

        #applying the classifier
        if classifier == "SVM":
            model = cl.applySVM(X_train_sel, y_train)
            model_raw = cl.applySVM(X_train_scaled, y_train, use_probabilities=False)
        elif classifier == "RandomForest":
            model = cl.applyRandForest(X_train_sel, y_train)
            model_raw = cl.applyRandForest(X_train_scaled, y_train, use_probabilities=False)
        elif classifier == "LogR":
            model = cl.applyLogR(X_train_sel, y_train)
            model_raw = cl.applyLogR(X_train_scaled, y_train)
        elif classifier == "DecisionTree":
            model = cl.applyDT(X_train_sel, y_train)
            model_raw = cl.applyDT(X_train_scaled, y_train)
        elif classifier == "MLP":
            model = cl.applyMLP(X_train_sel, y_train)
            model_raw = cl.applyMLP(X_train_scaled, y_train)

        y_pred = model.predict(X_test_sel)
        y_pred_raw = model_raw.predict(X_test_scaled)

        try:
            y_proba = model.predict_proba(X_test_sel)[:, 1]
        except:
            y_proba = None

        try:
            y_proba_raw = model.predict_proba(X_test_scaled)[:, 1]
        except:
            y_proba_raw = None

        perf = evaluate_performance(y_test, y_pred, y_proba, classifier_name=classifier, fold_idx=len(acc_scores) + 1, verbose=False)
        perf_raw = evaluate_performance(y_test, y_pred_raw, y_proba_raw, classifier_name=classifier, fold_idx=len(acc_scores) + 1, verbose=False)

        acc_scores.append(perf["accuracy"])
        mse_scores.append(mean_squared_error(y_test, y_pred))
        precision_scores.append(perf["precision"])
        recall_scores.append(perf["recall"])
        F1_scores.append(perf["f1"])
        AUC_scores.append(perf["auc"])

        if raw==True:
            perf_raw = evaluate_performance(y_test, y_pred_raw, y_proba_raw, classifier_name=classifier, fold_idx=len(acc_scores) + 1, verbose=False)
            # Raw performance
            acc_scores_raw.append(perf_raw["accuracy"])
            mse_scores_raw.append(mean_squared_error(y_test, y_pred))
            precision_scores_raw.append(perf_raw["precision"])
            recall_scores_raw.append(perf_raw["recall"])
            F1_scores_raw.append(perf_raw["f1"])
            AUC_scores_raw.append(perf_raw["auc"])

            avg_acc_raw = np.mean(acc_scores_raw)
            avg_mse_raw = np.mean(mse_scores_raw)
            avg_precision_raw = np.mean(precision_scores_raw)
            avg_recall_raw = np.mean(recall_scores_raw)
            avg_F1_raw = np.mean(F1_scores_raw)
            avg_AUC_raw = np.mean(AUC_scores_raw)

            print(f"Average accuracy raw: {avg_acc_raw}")
            print(f"Average mse raw: {avg_mse_raw}")
            print(f"Average precision raw: {avg_precision_raw}")
            print(f"Average recall raw: {avg_recall_raw}")
            print(f"Average F1 raw: {avg_F1_raw}")
            print(f"Average AUC raw: {avg_AUC_raw}")

    # Calculate averages (only if we have results)
    if acc_scores:
        # Get feature names for the last fold's selection
        if selected_features is not None:
            selected_feature_names = [feature_names[i] for i in selected_features 
                                    if i < len(feature_names)]
        else:
            selected_feature_names = list(feature_names)

    avg_acc = np.mean(acc_scores)
    avg_mse = np.mean(mse_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_F1 = np.mean(F1_scores)
    avg_AUC = np.mean(AUC_scores)

    print(f"Average accuracy: {avg_acc}")
    print(f"Average mse: {avg_mse}")
    print(f"Average precision: {avg_precision}")
    print(f"Average recall: {avg_recall}")
    print(f"Average F1: {avg_F1}")
    print(f"Average AUC: {avg_AUC}")

    avg_acc_raw = np.mean(acc_scores_raw)
    avg_mse_raw = np.mean(mse_scores_raw)

    return selected_features, selected_feature_names

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
    adj_values = subject_row.drop(columns=['subject_id', 'DX_GROUP', 'SEX', 'SITE_ID']).values.flatten()
    
    # Reshape the flattened array back into a square matrix
    adj_matrix = adj_values.reshape(matrix_size, matrix_size)
    
    # Plot the adjacency matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj_matrix, cmap="YlGnBu", annot=False, xticklabels=False, yticklabels=False)
    plt.title(f"Adjacency Matrix for Subject {subject_id}")
    plt.show()
    
def main():

    #Choose method: partial_corr_LF|partial_corr_glasso|pearson_corr_binary|pearson_corr|mutual_info|norm_laplacian|rlogspect
    X, y = load_file(sex='female', method='rlogspect')

    classifier = "SVM"  # Choose from SVM, RandomForest, LogR, DecisionTree, MLP

    X_train, X_test, y_train, y_test = train_and_evaluate(X, y, classifier)

    X_clustered = cluster.cluster(X_train, y_train, t=1)  # Clustering to select features
    print(f"Features selected by clustering({X_clustered.shape[1]}):", X_clustered)
    X_mRMR = mRMR(X_train, y_train, num_features_to_select=150)
    print(f"Features selected by mRMR({len(X_mRMR)}):", X_mRMR)
 
    #Cross-validation with feature selection
    selected_features_cv, selected_feature_names_cv = cross_validate_model(X, y, l1_logistic_regression, classifier, C=1.0, max_iter=1000)
    print("Cross-validated L1 Logistic Regression selected features:")
    print_selected_features(selected_features_cv, selected_feature_names_cv, print_feat=True)
    print("\n\n")

    selected_features_lars_cv, selected_feature_names_lars_cv = cross_validate_model(X, y, lars_lasso, classifier)
    print("Cross-validated LARS Lasso selected features:")
    print_selected_features(selected_features_lars_cv, selected_feature_names_lars_cv, print_feat=True)
    print("\n\n")

    selected_features_LAND_cv, selected_feature_names_LAND_cv = cross_validate_model(X, y, LAND, classifier)
    print("Cross-validated LAND selected features:")
    print_selected_features(selected_features_LAND_cv, selected_feature_names_LAND_cv, print_feat=True)
    print("\n\n")

    selected_features_hsiclasso_cv, selected_feature_names_hsiclasso_cv = cross_validate_model(X, y, hsic_lasso.hsic_lasso_forward_selection, classifier)
    print("Cross-validated HSIC Lasso selected features:")
    print_selected_features(selected_features_hsiclasso_cv, selected_feature_names_hsiclasso_cv, print_feat=True)
    print("\n\n")

    selected_features_mRMR_cv, selected_feature_names_mRMR_cv = cross_validate_model(X, y, mRMR, classifier, n_splits=5, num_features_to_select=50)
    print("Cross-validated mRMR selected features:")
    print_selected_features(selected_features_mRMR_cv, selected_feature_names_mRMR_cv, print_feat=True)
    print("\n\n")

    selected_features_permutation = failsafe_feature_selection(Perm_importance, X_train, y_train, classifier=classifier, select_features=X_clustered)
    print("Permutation Importance selected features clustered:")
    selected_feature_names_permutation = classify(X_train, X_test, y_train, y_test, selected_features_permutation, classifier)
    print_selected_features(selected_features_permutation, selected_feature_names_permutation, print_feat=True)
    print("\n\n")

    selected_features_permutation_mRMR = failsafe_feature_selection(Perm_importance, X_train, y_train, classifier=classifier, select_features=X_mRMR)
    print("Permutation Importance selected features mRMR")
    selected_features_names_perm_mRMR = classify(X_train, X_test, y_train, y_test, selected_features_permutation_mRMR, classifier)
    print_selected_features(selected_features_permutation_mRMR, selected_features_names_perm_mRMR, print_feat=True)
    print("\n\n")

    selected_features_sfs = failsafe_feature_selection(backwards_SFS, X_train, y_train, min_features=20, classifier=classifier, select_features=X_clustered, n_features_to_select=20)
    print("Sequential Feature Selection (SFS) selected features:")
    selected_features_names_sfs = classify(X_train, X_test, y_train, y_test, selected_features_sfs, classifier)
    print_selected_features(selected_features_sfs, selected_features_names_sfs, print_feat=True)
    print("\n\n")

    selected_features_sfs_mRMR = failsafe_feature_selection(backwards_SFS, X_train, y_train, min_features=20, classifier=classifier, select_features=X_mRMR, n_features_to_select=20)
    print("Sequential Feature Selection (SFS) selected features mRMR:")
    selected_features_names_sfs_mRMR = classify(X_train, X_test, y_train, y_test, selected_features_sfs_mRMR, classifier)
    print_selected_features(selected_features_sfs_mRMR, selected_features_names_sfs_mRMR, print_feat=True)
    print("\n\n")

    # X_train_scaled = fs.low_variance(X_train_scaled, threshold=0.01)
    # X_test_scaled = fs.low_variance(X_test_scaled, threshold=0.01)
    # selected_features_rfe = fs.backwards_SFS(X_train_scaled, y_train, 10, classifier)
    # acc, mse, selected_feature_names = classify(X, X_train_scaled, X_test_scaled, y_train, y_test, selected_features_rfe, classifier)
    # print("SFS selected features:")
    # print_selected_features(acc, mse, selected_features_rfe, selected_feature_names)
    # print("\n\n")

if __name__ == '__main__':
    fmri_data = main()


# %%
