from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from classification.src import classifiers as cl
from featureselection.src import feature_selection_methods as fs
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from featuredesign.graph_inference.AAL_test import multiset_feats, load_files

def load_file():
    #folder_path = r"C:\Users\guus\Python_map\AutismDetection-main\abide\female-cpac-filtnoglobal-aal" # Enter your local ABIDE dataset path
    data_arrays, file_paths, subject_ids, institude_names, metadata = load_files()

    full_df = multiset_feats(data_arrays)

    print("Merged feature+label shape:\n", full_df.shape)

    print(full_df)

    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the DataFrame

    X = full_df.drop(columns=['DX_GROUP', 'subject_id'])
    y = full_df['DX_GROUP'].map({1: 1, 2: 0})

    # Making sure the data is numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1,how='all')
    non_nan_ratio = X.notna().mean()
    X = X.loc[:, non_nan_ratio > 0.8]  # Keep columns with more than 50% non-NaN values
    # Making sure there is no 0 var data for the hsic algorithm
    X = X.loc[:, X.var() > 1e-6]

    # Fill rows with NaN values
    X= X.fillna(X.median())

    return X, y

def load_data():
    # Load with encoding fix
    data = pd.read_csv('ABIDEII_Composite_Phenotypic.csv', encoding='ISO-8859-1')

    # Drop rows where DX_GROUP is missing
    data = data[data['DX_GROUP'].notna()]

    # Separate X and y
    X = data.drop(columns=['DX_GROUP', 'PDD_DSM_IV_TR', 'SUB_ID', 'SITE_ID', 'AGE_AT_SCAN'])
    y = data['DX_GROUP']
    y = y.replace({1: 1, 2: 0})

    # Encode categorical
    X = pd.get_dummies(X, drop_first=True)

    #drop the columns where all values are missing before the imputer
    X = X.dropna(axis=1, how='all')

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return(X, y)

def train_and_evaluate(X, y):
    #splitting the data in train and test 0.8:0.2 respecively
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, stratify=y)

    #scale the data for the classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #applying the classifier to the total data
    model_raw = cl.applySVM(X_train_scaled, y_train)
    y_pred_raw = model_raw.predict(X_test_scaled)

    #finding mse and accuracy
    mse_raw = mean_squared_error(y_test, y_pred_raw)
    acc_raw = accuracy_score(y_test, y_pred_raw)
    print(classification_report(y_test, y_pred_raw, target_names=["Class 0", "Class 1"]))
    print('Confusion matrix:', confusion_matrix(y_test, y_pred_raw))
    print('Amount of features:', X_train.shape[1])

    #acc, mse, selected_feature_names = cross_validate_model(X, y, selected_features)
    print(f"Train/Test Accuracy raw: {acc_raw:.4f}, MSE: {mse_raw:.4f}")

    return X_train_scaled, X_test_scaled, y_train, y_test

def classify(X, X_train, X_test, y_train, y_test, selected_features, classifier):

    #changing the test and train data to the data selected by the feature selection tool
    selected_train_x = X_train[:, selected_features]
    selected_test_X = X_test[:, selected_features]

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
    y_pred = model.predict(selected_test_X)
    #params=bestSVM_RS(X_train, X_test, y_train, y_test, svcdefault=SVC())
    #finding mse and accuracy
    mse = mean_squared_error(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))
    print('Confusion matrix:', confusion_matrix(y_test, y_pred))

    #getting and printing the feature names
    feature_names = X.columns
    selected_feature_names = feature_names[selected_features]
    
    return acc, mse, selected_feature_names

def cross_validate_model(X, y, selected_features, classifier, n_splits=5):
    #K-Fold cross-validation evaluation.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores = []
    mse_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Select the features based on the selected indices 
        X_train_sel = X_train[:, selected_features]
        X_test_sel = X_test[:, selected_features]

        #Scaling the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        #applying the classifier
        if classifier == "SVM":
            model = cl.applySVM(X_train_scaled, y_train)
        elif classifier == "RandomForest":
            model = cl.applyRandForest(X_train_scaled, y_train)
        elif classifier == "LogR":
            model = cl.applyLogR(X_train_scaled, y_train)
        elif classifier == "DecisionTree":
            model = cl.applyDT(X_train_scaled, y_train)
        elif classifier == "MLP":
            model = cl.applyMLP(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)


        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc_scores.append(accuracy_score(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))

    #getting and printing the feature names
    feature_names = X_train.columns
    selected_feature_names = feature_names[selected_features]

    avg_acc = np.mean(acc_scores)
    avg_mse = np.mean(mse_scores)
    return avg_acc, avg_mse, selected_feature_names

def print_selected_features(acc, mse, selected_features, selected_feature_names):
    print("Selected features:", selected_features)
    print(f"Train/Test Accuracy: {acc:.4f}, MSE: {mse:.4f}")
    print(f"\nSelected feature names({len(selected_feature_names)}):")
    for name in selected_feature_names:
        print("-", name)

def main():

    X, y = load_file()

    classifier = "SVM"

    X_train_scaled, X_test_scaled, y_train, y_test = train_and_evaluate(X, y)

    selected_features_lars = fs.lars_lasso(X_train_scaled, y_train, alpha=0.1)
    acc, mse, selected_feature_names = classify(X, X_train_scaled, X_test_scaled, y_train, y_test, selected_features_lars, classifier)
    print("LARS Lasso selected features:")
    print_selected_features(acc, mse, selected_features_lars, selected_feature_names)
    print("\n\n")

    selected_features_lars = fs.LAND(X_train_scaled, y_train, lambda_reg=0.01)
    acc, mse, selected_feature_names = classify(X, X_train_scaled, X_test_scaled, y_train, y_test, selected_features_lars, classifier)
    print("LAND selected features:")
    print_selected_features(acc, mse, selected_features_lars, selected_feature_names)
    print("\n\n")

    X_train_scaled = fs.low_variance(X_train_scaled, threshold=0.01)
    X_test_scaled = fs.low_variance(X_test_scaled, threshold=0.01)
    selected_features_rfe = fs.backwards_SFS(X_train_scaled, y_train, 10, classifier)
    acc, mse, selected_feature_names = classify(X, X_train_scaled, X_test_scaled, y_train, y_test, selected_features_rfe, classifier)
    print("SFS selected features:")
    print_selected_features(acc, mse, selected_features_rfe, selected_feature_names)
    print("\n\n")

if __name__ == '__main__':
    main()
