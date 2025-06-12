#%%
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from classification.src import classifiers as cl, basicfeatureextraction
from featureselection.src.feature_selection_methods import *
from featureselection.src import cluster
from featureselection.src import Compute_HSIC_Lasso as hsic_lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from featuredesign.graph_inference.AAL_test import multiset_feats, load_files, adjacency_df
import glob
import cvxpy as cp
import seaborn as sns

def load_file(sex='all', method='pearson_corr', alpha=5):
    #folder_path = r"C:\Users\guus\Python_map\AutismDetection-main\abide\female-cpac-filtnoglobal-aal" # Enter your local ABIDE dataset path
    fmri_data, subject_ids, _, _ = load_files(sex=sex, max_files=800, site="NYU", shuffle=True, var_filt=True, ica=True)

    print(f"Final data: {len(fmri_data)} subjects")
    print(f"Final IDs: {len(subject_ids)}")

    full_df = adjacency_df(fmri_data, subject_ids, method = method, alpha = alpha)
    print("Merged feature+label shape:\n", full_df.shape)

    #print(full_df)
    
    subject_id_to_plot = '0051044'  # Change this to any valid subject ID
    #plot_adjacency_matrix(full_df, subject_id_to_plot)
    
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the DataFrame

    X = full_df.drop(columns=['DX_GROUP', 'subject_id', 'SEX'])
    y = full_df['DX_GROUP'].map({1: 1, 2: 0}) #1 ASD, 0 ALL

    # Making sure the data is numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1,how='all')
    non_nan_ratio = X.notna().mean()
    X = X.loc[:, non_nan_ratio > 0.8]  # Keep columns with more than 50% non-NaN values
    # Making sure there is no 0 var data for the hsic algorithm
    X = X.loc[:, X.var() > 1e-6]

    # NaN values are filled with the median of the column
    X= X.fillna(X.median())

    return X, y

def load_full_corr(sex='all', site_id=None):

    fc_female = basicfeatureextraction.extract_fc_features("abide/female-cpac-filtnoglobal-aal", "abide/Phenotypic_V1_0b_preprocessed1.csv")
    fc_male = basicfeatureextraction.extract_fc_features("abide/male-cpac-filtnoglobal-aal", "abide/Phenotypic_V1_0b_preprocessed1.csv")
    if sex == 'female':
        fc = fc_female
    elif sex == 'male':
        fc = fc_male
    elif sex == 'all':
        fc = pd.concat([fc_female, fc_male], axis=0, ignore_index=True)
    else:
        print("Use male, female or all as sex")

    if site_id is not None:
        fc = fc[fc['SITE_ID'] == site_id]

    fc = fc.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the DataFrame
    fc = fc.dropna(subset=['DX_GROUP'])

    X = fc.drop(columns=['DX_GROUP', 'SEX', 'SITE_ID', 'subject_id', 'AGE'])
    y = fc['DX_GROUP']

    # Making sure the data is numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1,how='all')
    non_nan_ratio = X.notna().mean()
    X = X.loc[:, non_nan_ratio > 0.8]  # Keep columns with more than 50% non-NaN values
    # Making sure there is no 0 var data for the hsic algorithm
    X = X.loc[:, X.var() > 1e-4]
    # NaN values are filled with the median of the column
    X = X.fillna(X.median())

    # Remove extremely correlated features
    X = correlation_filter(X, threshold=0.9)    
    print(f"After outlier removal: {X.shape}")
    # Remove extreme outliers
    X = remove_extreme_outliers(X, threshold=3.5)
    #print(f"After outlier removal: {X.shape}")

    # Apply feature transformations for better distributions
    X = apply_feature_transformations(X)
    #print(f"After transformation: {X.shape}")

    # Site effect correction
    if 'SITE_ID' in fc.columns:
        X = correct_site_effects(X, fc['SITE_ID'])
        #print(f"After site correction: {X.shape}")

    #print(f"X: {X}, y: {y}")

    return X, y

def load_dataframe(path='multi'):
    if path =='uni':
        folder_path = 'Feature_Dataframes/first_run'
    if path == 'multi':
        folder_path = 'Feature_Dataframes/second_run'
        #file_name = 'cpac_rois-aal_nogsr_filt_LADMM_direct_20ICA_graph_thr0.3.csv'
        file_name = 'cpac_rois-aal_nogsr_filt_LADMM_var_20ICA_graph_thr0.3.csv'
        #file_name = 'cpac_rois-aal_nogsr_filt_norm-laplacian_direct_20ICA_graph_thr0.3.csv'
        #file_name = 'cpac_rois-aal_nogsr_filt_norm-laplacian_glasso_20ICA_graph_thr0.3.csv'
        #file_name = 'cpac_rois-aal_nogsr_filt_norm-laplacian_ledoit_20ICA_graph_thr0.3.csv'
        #file_name = 'cpac_rois-aal_nogsr_filt_norm-laplacian_var_20ICA_graph_thr0.3.csv'

    file_path = os.path.join(folder_path, file_name)
    fc = pd.read_csv(file_path)
    #fc = pd.concat([pd.read_csv(file) for file in glob.glob(os.path.join(folder_path, '*.csv'))], ignore_index=True)

    fc = fc.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the DataFrame
    #fc = fc.dropna(subset=['DX_GROUP'])

    X = fc.drop(columns=['DX_GROUP', 'SEX', 'SITE_ID', 'subject_id', 'AGE_AT_SCAN'])
    y = fc['DX_GROUP']

    # Making sure the data is numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1,how='all')
    non_nan_ratio = X.notna().mean()
    X = X.loc[:, non_nan_ratio > 0.8]  # Keep columns with more than 50% non-NaN values
    # Making sure there is no 0 var data for the hsic algorithm
    X = X.loc[:, X.var() > 1e-4]
    # NaN values are filled with the median of the column
    X = X.fillna(X.median())
    print(f"shape dataframe: {X.shape}")
    # Remove extremely correlated features
    #X = correlation_filter(X, threshold=0.9)    
    #print(f"After correlation filter: {X.shape}")
    # Remove extreme outliers
    #X = remove_extreme_outliers(X, threshold=3.5)
    #print(f"After outlier removal: {X.shape}")

    # Apply feature transformations for better distributions
    #X = apply_feature_transformations(X)
    #print(f"After transformation: {X.shape}")

    # Site effect correction
    #if 'SITE_ID' in fc.columns:
    #    X = correct_site_effects(X, fc['SITE_ID'])
        #print(f"After site correction: {X.shape}")

    print(f"X: {X}, y: {y}")

    return X, y

def remove_extreme_outliers(X, threshold=3.5):
    """Remove extreme outliers using modified Z-score"""
    X_clean = X.copy()
    
    for col in X_clean.columns:
        # Calculate modified Z-score using median and MAD
        median = X_clean[col].median()
        mad = np.median(np.abs(X_clean[col] - median))
        if mad == 0:
            continue
        modified_z_scores = 0.6745 * (X_clean[col] - median) / mad
        
        # Cap values where |modified_z_score| > threshold
        outlier_mask = modified_z_scores > threshold
        X_clean.loc[outlier_mask, col] = median + threshold * mad / 0.6745
        
        outlier_mask = modified_z_scores < -threshold  
        X_clean.loc[outlier_mask, col] = median - threshold * mad / 0.6745
    
    return X_clean

def apply_feature_transformations(X):
    """Apply various transformations to improve feature distributions"""
    X_transformed = X.copy()
    
    for col in X_transformed.columns:
        data = X_transformed[col].values
        
        # Check skewness
        skewness = stats.skew(data)
        
        if abs(skewness) > 1.5:  # Highly skewed
            if skewness > 0:  # Right skewed
                # Try log transformation for positive skew
                if np.all(data > 0):
                    X_transformed[col] = np.log1p(data)
                else:
                    # Use Box-Cox like transformation
                    X_transformed[col] = np.sign(data) * np.log1p(np.abs(data))
            else:  # Left skewed
                # Use square transformation for negative skew
                X_transformed[col] = np.square(data)
    
    return X_transformed

def correct_site_effects(X, site_ids):
    """Correct for site effects using ComBat-like approach"""
    X_corrected = X.copy()
    
    # Group by site
    unique_sites = site_ids.unique()
    if len(unique_sites) <= 1:
        return X_corrected
    
    for col in X_corrected.columns:
        site_means = []
        site_vars = []
        
        # Calculate site-specific statistics
        for site in unique_sites:
            site_mask = site_ids == site
            site_data = X_corrected.loc[site_mask, col]
            site_means.append(site_data.mean())
            site_vars.append(site_data.var())
        
        # Apply correction
        overall_mean = X_corrected[col].mean()
        overall_var = X_corrected[col].var()
        
        for i, site in enumerate(unique_sites):
            site_mask = site_ids == site
            if site_vars[i] > 0:
                # Standardize and rescale
                X_corrected.loc[site_mask, col] = (
                    (X_corrected.loc[site_mask, col] - site_means[i]) / np.sqrt(site_vars[i])
                ) * np.sqrt(overall_var) + overall_mean
    
    return X_corrected

def correlation_filter(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_filtered = X.drop(columns=to_drop)
    print(f"Correlation filter: dropped {len(to_drop)} features.")
    return X_filtered

def evaluate_performance(y_true, y_pred, y_proba=None, show_plots=False, classifier_name="", fold_idx=None, verbose=True):
    # Compute basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    if verbose==True:
        print(f"\nPerformance Metrics ({classifier_name}):")
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
    num_feat = len(selected_features)
    print(f"Selected features ({num_feat}):", selected_features)
    #if print_feat==True:
    #    print(f"\nSelected feature names({len(selected_feature_names)}):")
    #    for name in selected_feature_names:
    #        print("-", name)

def train_and_evaluate(X, y, classifier):
    #splitting the data in train and test 0.8:0.2 respecively
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"y_train: {y_train}, y_test: {y_test}")

    #scale the data for the classifier
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if classifier == "SVM":
        model_raw = cl.applySVM(X_train_scaled, y_train)
    elif classifier == "RandomForest":
        model_raw = cl.applyRandForest(X_train_scaled, y_train)
    elif classifier == "LogR":
        model_raw = cl.applyLogR(X_train_scaled, y_train)
    elif classifier == "LDA":
        model_raw = cl.applyLDA(X_train_scaled, y_train)
    elif classifier == "KNN":
        model_raw = cl.applyKNN(X_train_scaled, y_train)
    else:
        print("Classifier not supported: choose from SVM, RandomForest, LogR, LDA or KNN")
    #applying the classifier to the total data
    model_raw = cl.applySVM(X_train, y_train)
    y_pred_raw = model_raw.predict(X_test)

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
    #print(classification_report(y_test, y_pred_raw, target_names=["Class 0", "Class 1"]))
    #print('Confusion matrix:', confusion_matrix(y_test, y_pred_raw))
    #print('Amount of features:', X_train.shape[1])

    #acc, mse, selected_feature_names = cross_validate_model(X, y, selected_features)
    print(f"Train/Test Accuracy raw: {acc_raw:.4f}, MSE: {mse_raw:.4f}, Precision: {precision_raw:.4f}, Recall: {recall_raw:.4f}, F1: {F1_raw:.4f}, AUC: {AUC_raw:.4f}")

    return X_train, X_test, y_train, y_test

def classify(X_train, X_test, y_train, y_test, selected_features, classifier, performance=True):

    #scale the data for the classifier
    scaler = StandardScaler()
    X_train_scaled = X_train #scaler.fit_transform(X_train)
    X_test_scaled = X_test #scaler.transform(X_test)

    if isinstance(X_train_scaled, pd.DataFrame):
        # If it's a DataFrame, use `.iloc[]` for indexing
        selected_train_x = X_train_scaled.iloc[:, selected_features]
        selected_test_x = X_test_scaled.iloc[:, selected_features]
    else:
        # If it's a numpy array, use standard array indexing
        selected_train_x = X_train_scaled[:, selected_features]
        selected_test_x = X_test_scaled[:, selected_features]

    if classifier == "SVM":
        model = cl.applySVM(selected_train_x, y_train)
    elif classifier == "RandomForest":
        model = cl.applyRandForest(selected_train_x, y_train)
    elif classifier == "LogR":
        model = cl.applyLogR(selected_train_x, y_train)
    elif classifier == "LDA":
        model = cl.applyLDA(selected_train_x, y_train)
    elif classifier == "KNN":
        model = cl.applyKNN(selected_train_x, y_train)
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

def cross_validate_model(X, y, feature_selection, classifier, raw=True, return_metrics=False, n_splits=5, **feature_selection_kwargs):
    #K-Fold cross-validation evaluation.
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) #shuffle=True, random_state=42
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
    fold_metrics = []

    if classifier is not Perm_importance or backwards_SFS:
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

    for train_idx, test_idx in kf.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        #Scaling the data
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if feature_selection is not None:
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
        else:
            X_train_sel = X_train_scaled
            X_test_sel = X_test_scaled

        #applying the classifier
        if classifier == "SVM":
            model = cl.applySVM(X_train_sel, y_train)
            model_raw = cl.applySVM(X_train_scaled, y_train)
        elif classifier == "RandomForest":
            model = cl.applyRandForest(X_train_sel, y_train)
            model_raw = cl.applyRandForest(X_train_scaled, y_train)
        elif classifier == "LogR":
            model = cl.applyLogR(X_train_sel, y_train)
            model_raw = cl.applyLogR(X_train_scaled, y_train)
        elif classifier == "DT":
            model = cl.applyDT(X_train_sel, y_train)
            model_raw = cl.applyDT(X_train_scaled, y_train)
        elif classifier == "MLP":
            model = cl.applyMLP(X_train_sel, y_train)
            model_raw = cl.applyMLP(X_train_scaled, y_train)
        elif classifier == "LDA":
            model = cl.applyLDA(X_train_sel, y_train)
            model_raw = cl.applyLDA(X_train_scaled, y_train)
        elif classifier == "KNN":
            model = cl.applyKNN(X_train_sel, y_train)
            model_raw = cl.applyKNN(X_train_scaled, y_train)

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

        acc_scores.append(perf["accuracy"] if perf["accuracy"] is not None else 0.0)
        mse_scores.append(mean_squared_error(y_test, y_pred))
        precision_scores.append(perf["precision"] if perf["precision"] is not None else 0.0)
        recall_scores.append(perf["recall"] if perf["recall"] is not None else 0.0)
        F1_scores.append(perf["f1"] if perf["f1"] is not None else 0.0)
        AUC_scores.append(perf["auc"] if perf["auc"] is not None else 0.0)

        fold_metrics.append({
            "accuracy": acc_scores,
            "precision": precision_scores,
            "recall": recall_scores,
            "f1_score": F1_scores,
            "auroc": AUC_scores
        })

        print(classifier)
        #print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))
        #print('Confusion matrix:', confusion_matrix(y_test, y_pred))

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
            avg_AUC_raw = np.mean([score for score in AUC_scores_raw if score is not None])

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
    """
    selected_features = failsafe_feature_selection(feature_selection, X, y, classifier=classifier, **feature_selection_kwargs)
    X_selected = X[:, selected_features]

    model = select_model(classifier) 

    acc_scores = cross_val_score(model, X_selected, y, cv=kf, scoring='accuracy')
    mse_scores = cross_val_score(model, X_selected, y, cv=kf, scoring='neg_mean_squared_error')
    precision_scores = cross_val_score(model, X_selected, y, cv=kf, scoring='precision')
    recall_scores = cross_val_score(model, X_selected, y, cv=kf, scoring='recall')
    F1_scores = cross_val_score(model, X_selected, y, cv=kf, scoring='f1')
    AUC_scores = cross_val_score(model, X_selected, y, cv=kf, scoring='roc_auc')
    """

    avg_acc = np.mean(acc_scores)
    avg_mse = np.mean(mse_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_F1 = np.mean(F1_scores)
    avg_AUC = np.mean(AUC_scores)

    avg_metrics = {
        "accuracy": avg_acc,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_F1,
        "auroc": avg_AUC,
        "sensitivity": avg_recall  # sensitivity == recall in binary classification

    }

    print(f"\nMean performance Metrics ({classifier}), ({feature_selection}):")
    print(f"Mean performance Metrics ({classifier}), ({feature_selection}):")
    print(f"  Accuracy:  {avg_acc:.4f}")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f}")
    print(f"  F1 Score:  {avg_F1:.4f}")
    if avg_AUC is not None:
        print(f"  AUC:       {avg_AUC:.4f}")

    if raw==True:
        acc_scores_raw = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        mse_scores_raw = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        precision_scores_raw = cross_val_score(model, X, y, cv=kf, scoring='precision')
        recall_scores_raw = cross_val_score(model, X, y, cv=kf, scoring='recall')
        F1_scores_raw = cross_val_score(model, X, y, cv=kf, scoring='f1')
        AUC_scores_raw = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')

        avg_acc_raw = np.mean(acc_scores_raw)
        avg_mse_raw = np.mean(mse_scores_raw)
        avg_precision_raw = np.mean(precision_scores_raw)
        avg_recall_raw = np.mean(recall_scores_raw)
        avg_F1_raw = np.mean(F1_scores_raw)
        avg_AUC_raw = np.mean(AUC_scores_raw)

        print(f"\nPerformance Metrics raw ({classifier}):")
        print(f"Performance Metrics raw ({classifier}):")
        print(f"  Accuracy:  {avg_acc_raw:.4f}")
        print(f"  Precision: {avg_precision_raw:.4f}")
        print(f"  Recall:    {avg_recall_raw:.4f}")
        print(f"  F1 Score:  {avg_F1_raw:.4f}")
        if avg_AUC is not None:
            print(f"  AUC:       {avg_AUC_raw:.4f}")

    if return_metrics:
        return selected_features, selected_feature_names, avg_metrics, fold_metrics
    else:
        selected_features, selected_feature_names

def select_model(classifier):
    # Determine the model based on the classifier name
    if classifier == "SVM":
        model = SVC(kernel='linear')
    elif classifier == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    elif classifier == "LogR":
        model = LogisticRegression(random_state=42)
    elif classifier == "DT":
        model = DecisionTreeClassifier(random_state=42)
    elif classifier == "MLP":
        model = MLPClassifier(random_state=42)
    elif classifier == "LDA":
        model = LinearDiscriminantAnalysis()
    elif classifier == "KNN":
        model = KNeighborsClassifier()
    else:
        raise ValueError("Unsupported classifier type, choose SVM, RandomForest, DT, MLP, LogR, LDA or KNN")
    
    return model

def main(classifier="LogR"):

    #Choose method: partial_corr_LF|partial_corr_glasso|pearson_corr_binary|pearson_corr|mutual_info|norm_laplacian|rlogspect
    #X, y = load_file(sex='all', method='pearson_corr', max_files=None)
    X, y = load_dataframe()

    # Choose from SVM, RandomForest, LogR, LDA, KNN

    X_train, X_test, y_train, y_test = train_and_evaluate(X, y, classifier)

    X_clustered = cluster.cluster(X_train, y_train, t=3)  # Clustering to select features
    X_clustered_big = cluster.cluster(X, y, t=1)
    print(f"Features selected by clustering({X_clustered.shape[1]}):", X_clustered)
    print(f"features selected by clustering big({X_clustered_big.shape[1]})")
    X_mRMR = mRMR(X_train, y_train, classifier, num_features_to_select=100)
    X_mRMR_big = mRMR(X, y, classifier, num_features_to_select=300)
    print(f"Features selected by mRMR({len(X_mRMR)}):", X_mRMR)
    print(f"features selected by mRMR big({len(X_mRMR_big)})")
    
    alpha_results_lasso = alpha_lasso_selection(X, y, classifier) #0.002812 with 98 features
    best_alpha = alpha_results_lasso['best_alpha']
    print(f"alpha lasso: {best_alpha}")
    num_feat_hsic = hsiclasso(X, y, classifier, verbose=False)
    print(f"num feat hsic: {num_feat_hsic}")
    #best_alpha = 0.002812
    #Cross-validation with feature selection
    selected_features_lasso_cv, selected_feature_names_lasso_cv = cross_validate_model(X, y, Lasso_selection, classifier, alpha=best_alpha, max_iter=2000)
    print("Cross-validated L1 Logistic Regression selected features:")
    print_selected_features(selected_features_lasso_cv, selected_feature_names_lasso_cv, print_feat=True)
    print("\n\n")

    #Cross-validation with feature selection
    #selected_features_lasso_cl, selected_feature_names_lasso_cl = cross_validate_model(X, y, Lasso_selection, classifier, alpha=best_alpha, max_iter=2000, selected_features=X_clustered_big)
    #print("Cross-validated L1 Logistic Regression selected features:")
    #print_selected_features(selected_features_lasso_cl, selected_feature_names_lasso_cl, print_feat=True)
    #print("\n\n")

    #Cross-validation with feature selection
    #selected_features_lasso_mr, selected_feature_names_lasso_mr = cross_validate_model(X, y, Lasso_selection, classifier, alpha=best_alpha, max_iter=2000, select_features=X_mRMR_big)
    #print("Cross-validated L1 Logistic Regression selected features:")
    #print_selected_features(selected_features_lasso_mr, selected_feature_names_lasso_mr, print_feat=True)
    #print("\n\n")

    num_feat_hsic = hsiclasso(X, y, classifier, verbose=False)
    selected_features_hsiclasso_cv, selected_feature_names_hsiclasso_cv = cross_validate_model(X, y, hsiclasso, classifier, num_feat=98) #92,
    print("Cross-validated HSIC Lasso selected features:")
    print_selected_features(selected_features_hsiclasso_cv, selected_feature_names_hsiclasso_cv, print_feat=True)
    print("\n\n")

    #selected_features_lars_cv, selected_feature_names_lars_cv = cross_validate_model(X, y, lars_lasso, classifier)
    #print("Cross-validated LARS Lasso selected features:")
    #print_selected_features(selected_features_lars_cv, selected_feature_names_lars_cv, print_feat=True)
    #print("\n\n")

    #selected_features_LAND_cv, selected_feature_names_LAND_cv = cross_validate_model(X, y, LAND, classifier)
    #print("Cross-validated LAND selected features:")
    #print_selected_features(selected_features_LAND_cv, selected_feature_names_LAND_cv, print_feat=True)
    #print("\n\n")

    selected_features_mRMR_cv, selected_feature_names_mRMR_cv = cross_validate_model(X, y, mRMR, classifier, n_splits=5, num_features_to_select=200)
    print("Cross-validated mRMR selected features:")
    print_selected_features(selected_features_mRMR_cv, selected_feature_names_mRMR_cv, print_feat=True)
    print("\n\n")

    selected_features_reliefF_cv, selected_feature_names_reliefF_cv = cross_validate_model(X, y, reliefF_, classifier, n_splits=5, num_features_to_select=200)
    print("Cross-validated reliefF selected features:")
    print_selected_features(selected_features_reliefF_cv, selected_feature_names_reliefF_cv, print_feat=True)
    print("\n\n")

    #selected_features_perm_cl, selected_feature_names_perm_cl = cross_validate_model(X, y, Perm_importance, classifier, n_splits=5, select_features=X_clustered)
    #print("Cross-validated mRMR selected features:")
    #print_selected_features(selected_features_perm_cl, selected_feature_names_perm_cl, print_feat=True)
    #print("\n\n")

    #selected_features_perm_cl, selected_feature_names_perm_cl = cross_validate_model(X, y, Perm_importance, classifier, n_splits=5, select_features=X_mRMR)
    #print("Cross-validated mRMR selected features:")
    #print_selected_features(selected_features_perm_cl, selected_feature_names_perm_cl, print_feat=True)
    #print("\n\n")

    selected_features_sfs = failsafe_feature_selection(backwards_SFS, X_train, y_train, min_features=10, classifier=classifier, select_features=X_clustered, n_features_to_select=20)
    print("Sequential Feature Selection (SFS) selected features:")
    selected_features_names_sfs = classify(X_train, X_test, y_train, y_test, selected_features_sfs, classifier)
    print_selected_features(selected_features_sfs, selected_features_names_sfs, print_feat=True)
    print("\n\n")

    selected_features_sfs_mRMR = failsafe_feature_selection(backwards_SFS, X_train, y_train, min_features=10, classifier=classifier, select_features=X_mRMR, n_features_to_select=20)
    print("Sequential Feature Selection (SFS) selected features mRMR:")
    selected_features_names_sfs_mRMR = classify(X_train, X_test, y_train, y_test, selected_features_sfs_mRMR, classifier)
    print_selected_features(selected_features_sfs_mRMR, selected_features_names_sfs_mRMR, print_feat=True)

    selected_features_permutation_cl = failsafe_feature_selection(Perm_importance, X_train, y_train, classifier=classifier, select_features=X_clustered)
    print("Permutation Importance selected features mRMR")
    selected_features_names_perm_cl = classify(X_train, X_test, y_train, y_test, selected_features_permutation_cl, classifier)
    print_selected_features(selected_features_permutation_cl, selected_features_names_perm_cl, print_feat=True)
    print("\n\n")

    selected_features_permutation_mRMR = failsafe_feature_selection(Perm_importance, X_train, y_train, classifier=classifier, select_features=X_mRMR)
    print("Permutation Importance selected features mRMR")
    selected_features_names_perm_mRMR = classify(X_train, X_test, y_train, y_test, selected_features_permutation_mRMR, classifier)
    print_selected_features(selected_features_permutation_mRMR, selected_features_names_perm_mRMR, print_feat=True)
    print("\n\n")

    # X_train_scaled = fs.low_variance(X_train_scaled, threshold=0.01)
    # X_test_scaled = fs.low_variance(X_test_scaled, threshold=0.01)
    # selected_features_rfe = fs.backwards_SFS(X_train_scaled, y_train, 10, classifier)
    # acc, mse, selected_feature_names = classify(X, X_train_scaled, X_test_scaled, y_train, y_test, selected_features_rfe, classifier)
    # print("SFS selected features:")
    # print_selected_features(acc, mse, selected_features_rfe, selected_feature_names)
    # print("\n\n")


if __name__ == '__main__':
     main()


# %%
