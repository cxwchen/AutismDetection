from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from classification.src import classifiers as cl, basicfeatureextraction
from featureselection.src.feature_selection_methods import *
from featureselection.src import cluster
from featureselection.src import Compute_HSIC_Lasso as hsic_lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from featuredesign.graph_inference.AAL_test import multiset_feats, load_files
import os,json
from featuredesign.graph_inference.AAL_test import load_files,multiset_feats, ica_smith, group_ica

from nilearn.datasets import fetch_abide_pcp
#from nilearn.connectome import ConnectivityMeasure
from dotenv import load_dotenv
from nilearn.datasets import fetch_abide_pcp
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
        output_dir = Path.home() / 'Documents' / 'abide_test'
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
            #data_list = ica_smith(data_list, n_components=n_components)  # ICA
            _,data_list = group_ica(data_list, n_components=n_components)

        result_df = multiset_feats(
            data_list=data_list,
            subject_ids=subject_ids,
            inf_method=inf_method,
            cov_method=cov_method,
            thresh=thresh,
            n_jobs=-1,
            feats=feats
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


def evaluate_performance(y_true, y_pred, y_proba=None, show_plots=False, classifier_name="", fold_idx=None,
                         verbose=True):
    # Compute basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    if verbose == True:
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
    if print_feat == True:
        print(f"\nSelected feature names({len(selected_feature_names)}):")
        for name in selected_feature_names:
            print("-", name)


def train_and_evaluate(X, y, classifier):
    # splitting the data in train and test 0.8:0.2 respecively
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, stratify=y)

    # scale the data for the classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
    # applying the classifier to the total data
    y_pred_raw = model_raw.predict(X_test_scaled)

    try:
        y_proba_raw = model_raw.predict_proba(X_test_scaled)[:, 1]
    except:
        y_proba_raw = None

    # finding mse and accuracy
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

    # acc, mse, selected_feature_names = cross_validate_model(X, y, selected_features)
    print(
        f"Train/Test Accuracy raw: {acc_raw:.4f}, MSE: {mse_raw:.4f}, Precision: {precision_raw:.4f}, Recall: {recall_raw:.4f}, F1: {F1_raw:.4f}, AUC: {AUC_raw:.4f}")

    return acc_raw, F1_raw, AUC_raw, X_train, X_test, y_train, y_test


def classify(X_train, X_test, y_train, y_test, selected_features, classifier, performance=True):
    # scale the data for the classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
    elif classifier == "DecisionTree":
        model = cl.applyDT(selected_train_x, y_train)
    elif classifier == "MLP":
        model = cl.applyMLP(selected_train_x, y_train)
    else:
        print("Classifier not supported: choose from SVM, RandomForest, LogR, DecisionTree or MLP")

    # applying the classifier to the selected data
    y_pred = model.predict(selected_test_x)
    # params=bestSVM_RS(X_train, X_test, y_train, y_test, svcdefault=SVC())
    # finding mse and accuracy

    # Predict probabilities if supported
    try:
        y_proba = model.predict(selected_test_x) if hasattr(model, "predict_proba") else None
        if y_proba is not None:
            y_proba = model.predict_proba(selected_test_x)[:, 1]
    except:
        y_proba = None
    if performance == True:
        evaluate_performance(y_test, y_pred, y_proba, classifier_name=classifier)
    # getting and printing the feature names
    feature_names = X_train.columns
    selected_feature_names = feature_names[selected_features]

    return selected_feature_names


def cross_validate_model(X, y, feature_selection, classifier, raw=True, n_splits=5, **feature_selection_kwargs):
    # K-Fold cross-validation evaluation.
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
    AUC_scores_raw = []

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

        # Scaling the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        selected_features = failsafe_feature_selection(feature_selection, X_train_scaled, y_train,
                                                       classifier=classifier, **feature_selection_kwargs)

        # Ensure selected_features is a list of valid indices
        if not isinstance(selected_features, (list, np.ndarray)):
            selected_features = [selected_features] if selected_features is not None else []

        selected_features = [int(idx) for idx in selected_features if
                             isinstance(idx, (int, np.integer)) and 0 <= idx < X_train.shape[1]]

        if not selected_features:
            # Fallback to all features if selection fails
            selected_features = list(range(X_train.shape[1]))

        # Select the features based on the selected indices
        X_train_sel = X_train_scaled[:, selected_features]
        X_test_sel = X_test_scaled[:, selected_features]

        # applying the classifier
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

        perf = evaluate_performance(y_test, y_pred, y_proba, classifier_name=classifier, fold_idx=len(acc_scores) + 1,
                                    verbose=False)

        acc_scores.append(perf["accuracy"] if perf["accuracy"] is not None else 0.0)
        mse_scores.append(mean_squared_error(y_test, y_pred))
        precision_scores.append(perf["precision"] if perf["precision"] is not None else 0.0)
        recall_scores.append(perf["recall"] if perf["recall"] is not None else 0.0)
        F1_scores.append(perf["f1"] if perf["f1"] is not None else 0.0)
        AUC_scores.append(perf["auc"] if perf["auc"] is not None else 0.0)

        if raw == True:
            perf_raw = evaluate_performance(y_test, y_pred_raw, y_proba_raw, classifier_name=classifier,
                                            fold_idx=len(acc_scores) + 1, verbose=False)
            # Raw performance
            acc_scores_raw.append(perf_raw["accuracy"] if perf_raw["accuracy"] is not None else 0.0)
            mse_scores_raw.append(mean_squared_error(y_test, y_pred_raw))
            precision_scores_raw.append(perf_raw["precision"] if perf_raw["precision"] is not None else 0.0)
            recall_scores_raw.append(perf_raw["recall"] if perf_raw["recall"] is not None else 0.0)
            F1_scores_raw.append(perf_raw["f1"] if perf_raw["f1"] is not None else 0.0)
            AUC_scores_raw.append(perf_raw["auc"] if perf_raw["auc"] is not None else 0.0)

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

    if raw == True:
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

    avg_acc_raw = np.mean(acc_scores_raw)
    avg_mse_raw = np.mean(mse_scores_raw)

    return selected_features, selected_feature_names


def read_feats(df_out):
    full_df = df_out.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the DataFrame

    X = full_df.drop(columns=['DX_GROUP', 'SEX', 'SITE_ID', 'subject_id'])
    y = full_df['DX_GROUP'].map({1: 1, 2: 0})

    # Making sure the data is numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all')
    non_nan_ratio = X.notna().mean()
    X = X.loc[:, non_nan_ratio > 0.5]  # Keep columns with more than 50% non-NaN values
    # Making sure there is no 0 var data for the hsic algorithm
    X = X.loc[:, X.var() > 1e-6]

    # NaN values are filled with the median of the column
    X = X.fillna(X.median())
    return X, y

def run_all_combinations():
    inf_methods = ['sample_cov', 'partial_corr', 'pearson_corr',
                   'mutual_info', 'gr_causality', 'rlogspect']
    cov_methods = ['direct', 'ledoit', 'glasso', 'window', 'var', 'nvar']
    inf_methods_requiring_cov = ['sample_cov', 'partial_corr', 'rlogspect']
    n_components = 20

    results_dir = os.path.expanduser("~/Documents/abide_test/accuracies")
    os.makedirs(results_dir, exist_ok=True)

    for inf in inf_methods:
        if inf in inf_methods_requiring_cov:
            # Try all combinations with cov_methods
            for cov in cov_methods:
                try:
                    print(f"Running inf_method={inf}, cov_method={cov}")
                    df_out, _ = process_feats(feats='graph', inf_method=inf, cov_method=cov, n_components=n_components)
                    print("df_out:\n", df_out)
                    X, y = read_feats(df_out)
                    acc, f1, auc, *_ = train_and_evaluate(X, y, classifier="SVM")

                    # Save results
                    result = {
                        "inf_method": inf,
                        "cov_method": cov,
                        "accuracy": acc,
                        "f1": f1,
                        "auc": auc
                    }
                    filename = f"{inf}_{cov}_{n_components}-IC.json"
                    with open(os.path.join(results_dir, filename), 'w') as f:
                        json.dump(result, f, indent=4)
                except Exception as e:
                    print(f"Failed for {inf} + {cov}: {e}")
        else:
            # These inf_methods do not use covariance
            try:
                print(f"Running inf_method={inf} (no cov_method)")
                df_out, _ = process_feats(feats='graph', inf_method=inf, cov_method=None)
                X, y = read_feats(df_out)
                acc, f1, auc, *_ = train_and_evaluate(X, y, classifier="SVM")

                result = {
                    "inf_method": inf,
                    "cov_method": None,
                    "accuracy": acc,
                    "f1": f1,
                    "auc": auc
                }
                filename = f"{inf}_nocov_{n_components}-IC.json"
                with open(os.path.join(results_dir, filename), 'w') as f:
                    json.dump(result, f, indent=4)
            except Exception as e:
                print(f"Failed for {inf} (no cov_method): {e}")
'''
    print("timeseries_shape: ", time_series[0].shape)

    X, y = read_feats(df_out)

    classifier = "SVM"  # Choose from SVM, RandomForest, LogR, DecisionTree, MLP

    X_train, X_test, y_train, y_test = train_and_evaluate(X, y, classifier)

    #print("df_out:\n", df_out)

    selected_features_hsiclasso_cv, selected_feature_names_hsiclasso_cv = cross_validate_model(X, y, hsiclasso,
                                                                                               classifier)
    print("Cross-validated HSIC Lasso selected features:")
    print_selected_features(selected_features_hsiclasso_cv, selected_feature_names_hsiclasso_cv, print_feat=True)
    print("\n\n")
'''
def main(run_full_grid=False):
    if run_full_grid:
        run_all_combinations()
    else:
        # Single experiment run
        df_out, time_series = process_feats(
            pipeline='cpac',
            site='NYU',
            gsr=True,
            filt=True,
            feats='graph',
            n_components=20,
            inf_method='partial_corr',
            cov_method='direct'
        )
        print("timeseries_shape: ", time_series[0].shape)

        X, y = read_feats(df_out)
        classifier = "SVM"
        acc, f1, auc, X_train, X_test, y_train, y_test = train_and_evaluate(X, y, classifier)

        print(f"Single run completed - Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

if __name__ == "__main__":
    main(run_full_grid=True)  # Set to False if you want a single test