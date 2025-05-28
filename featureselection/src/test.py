from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from classification.src import classifiers as cl
from featureselection.src import feature_selection_methods as fs
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from featuredesign.graph_inference.AAL_test import load_files,multiset_feats
from Pipeline import load_data,train_and_evaluate,classify,cross_validate_model,print_selected_features

def process(data):
    # Drop rows where DX_GROUP is missing
    data = data[data['DX_GROUP'].notna()]

    # Separate X and y
    X = data.drop(columns=['DX_GROUP', 'subject_id', 'SITE_ID', 'SEX'], errors='ignore')
    y = data['DX_GROUP'].replace({1: 1, 2: 0})

    # 🔍 Separate scalar vs array columns
    array_cols = [col for col in X.columns if X[col].apply(lambda x: isinstance(x, (np.ndarray, list))).any()]
    scalar_cols = [col for col in X.columns if col not in array_cols]

    X_scalar = X[scalar_cols]

    # Encode categorical
    X_scalar = pd.get_dummies(X_scalar, drop_first=True)

    # Drop columns where all values are missing
    X_scalar = X_scalar.dropna(axis=1, how='all')

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_scalar = pd.DataFrame(imputer.fit_transform(X_scalar), columns=X_scalar.columns)

    return X_scalar, y

def main():

    fmri_data, subject_ids, _, _ = load_files(sex='all', max_files=800, shuffle=True, var_filt=True, ica=True)
    print(f"Final data: {len(fmri_data)} subjects")
    print(f"Final IDs: {len(subject_ids)}")

    df_out = multiset_feats(fmri_data, subject_ids, method='rlogspect')

    print("df_out:\n", df_out)

    X, y = process(df_out)

    classifier = "SVM"

    X_train_scaled, X_test_scaled, y_train, y_test = train_and_evaluate(X, y)

    selected_features_lars = fs.lars_lasso(X_train_scaled, y_train, alpha=0.1)
    acc, mse, selected_feature_names = classify(X, X_train_scaled, X_test_scaled, y_train, y_test, selected_features_lars, classifier)
    print("LARS Lasso selected features:")
    print_selected_features(acc, mse, selected_features_lars, selected_feature_names)
    print("\n\n")

    selected_features_rfe = fs.backwards_SFS(X_train_scaled, y_train, 10, classifier)
    acc, mse, selected_feature_names = classify(X, X_train_scaled, X_test_scaled, y_train, y_test, selected_features_rfe, classifier)
    print("SFS selected features:")
    print_selected_features(acc, mse, selected_features_rfe, selected_feature_names)
    print("\n\n")

if __name__ == '__main__':
    main()