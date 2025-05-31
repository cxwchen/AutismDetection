from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from classification.src import classifiers as cl
from featureselection.src import feature_selection_methods as fs
from featureselection.src import cluster
from featureselection.src import Compute_HSIC_Lasso as hsic_lasso
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

def train_and_evaluate(X, y, classifier):
    #splitting the data in train and test 0.8:0.2 respecively
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, stratify=y)

    #scale the data for the classifier
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
    #applying the classifier to the total data
    y_pred_raw = model_raw.predict(X_test_scaled)

    #finding mse and accuracy
    mse_raw = mean_squared_error(y_test, y_pred_raw)
    acc_raw = accuracy_score(y_test, y_pred_raw)
    print(classification_report(y_test, y_pred_raw, target_names=["Class 0", "Class 1"]))
    print('Confusion matrix:', confusion_matrix(y_test, y_pred_raw))
    print('Amount of features:', X_train.shape[1])

    #acc, mse, selected_feature_names = cross_validate_model(X, y, selected_features)
    print(f"Train/Test Accuracy raw: {acc_raw:.4f}, MSE: {mse_raw:.4f}")

    return X_train, X_test, y_train, y_test

def classify(X_train, X_test, y_train, y_test, selected_features, classifier):

    selected_train_x = X_train.iloc[:, selected_features]
    selected_test_x = X_test.iloc[:, selected_features]

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
    mse = mean_squared_error(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))
    print('Confusion matrix:', confusion_matrix(y_test, y_pred))

    #getting and printing the feature names
    feature_names = X_train.columns
    selected_feature_names = feature_names[selected_features]
    
    return acc, mse, selected_feature_names

def cross_validate_model(X, y, feature_selection, classifier, n_splits=5):
    #K-Fold cross-validation evaluation.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores = []
    mse_scores = []
    acc_scores_raw = []
    mse_scores_raw = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if feature_selection == "low_variance":
            # Apply low variance feature selection
            X_train, selected_features = fs.low_variance(X_train, threshold=0.5)
        elif feature_selection == "Perm_importance":
            # Apply Permutation Importance feature selection
            selected_features = fs.Perm_importance(X_train, y_train, classifier=classifier)
        elif feature_selection == "lars_lasso":
            # Apply LARS Lasso feature selection
            selected_features = fs.lars_lasso(X_train, y_train, alpha=0.1)
        elif feature_selection == "LAND":
            # Apply LAND feature selection
            selected_features = fs.LAND(X_train, y_train, lambda_reg=0.01)
        elif feature_selection == "SFS":
            # Apply Sequential Feature Selection (SFS)
            X_train_var, selected_features = fs.backwards_SFS(X_train, y_train, 10, classifier)
        else:
            raise ValueError("Unsupported feature selection method")
        

        # Select the features based on the selected indices 
        X_train_sel = X_train.iloc[:, selected_features]
        X_test_sel = X_test.iloc[:, selected_features]

        #Scaling the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        #applying the classifier
        if classifier == "SVM":
            model = cl.applySVM(X_train_scaled, y_train)
            model_raw = cl.applySVM(X_train_scaled, y_train, use_probabilities=False)
        elif classifier == "RandomForest":
            model = cl.applyRandForest(X_train_scaled, y_train)
            model_raw = cl.applyRandForest(X_train_scaled, y_train, use_probabilities=False)
        elif classifier == "LogR":
            model = cl.applyLogR(X_train_scaled, y_train)
            model_raw = cl.applyLogR(X_train_scaled, y_train)
        elif classifier == "DecisionTree":
            model = cl.applyDT(X_train_scaled, y_train)
            model_raw = cl.applyDT(X_train_scaled, y_train)
        elif classifier == "MLP":
            model = cl.applyMLP(X_train_scaled, y_train)
            model_raw = cl.applyMLP(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_pred_raw = model_raw.predict(X_test_scaled)

        acc_scores_raw.append(accuracy_score(y_test, y_pred_raw))
        mse_scores_raw.append(mean_squared_error(y_test, y_pred_raw))

        print("Accuracy raw:", acc_scores_raw[-1])
        print("MSE raw:", mse_scores_raw[-1])

        acc_scores.append(accuracy_score(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))

    #getting and printing the feature names
    feature_names = X_train.columns
    selected_feature_names = feature_names[selected_features]

    avg_acc = np.mean(acc_scores)
    avg_mse = np.mean(mse_scores)
    return avg_acc, avg_mse, selected_features, selected_feature_names

def print_selected_features(acc, mse, selected_features, selected_feature_names):
    print("Selected features:", selected_features)
    print(f"Train/Test Accuracy: {acc:.4f}, MSE: {mse:.4f}")
    #print(f"\nSelected feature names({len(selected_feature_names)}):")
    #for name in selected_feature_names:
    #    print("-", name)

def main():

    X, y = load_file()

    classifier = "RandomForest"  # Choose from SVM, RandomForest, LogR, DecisionTree, MLP

    X_train, X_test, y_train, y_test = train_and_evaluate(X, y, classifier)

    X_clustered = cluster.cluster(X_train, y_train, t=2)  # Clustering to select features
    print("Number of features after clustering:", X_clustered.shape[1])
    X_mRMR = fs.mRMR(X_train, y_train, num_features_to_select=150)

    selected_features_logreg = fs.l1_logistic_regression(X_train, y_train, C=0.1)
    acc_logreg, mse_logreg, selected_feature_names_logreg = classify(X_train, X_test, y_train, y_test, selected_features_logreg, classifier)
    print("L1 Logistic Regression selected features:")
    print_selected_features(acc_logreg, mse_logreg, selected_features_logreg, selected_feature_names_logreg)
    print("\n\n")

    selected_features_lars = fs.lars_lasso(X_train, y_train, alpha=0.1)
    acc_lars, mse_lars, selected_feature_names_lars = classify(X_train, X_test, y_train, y_test, selected_features_lars, classifier)
    print("LARS Lasso selected features:")
    print_selected_features(acc_lars, mse_lars, selected_features_lars, selected_feature_names_lars)
    print("\n\n")

    selected_features_LAND = fs.LAND(X_train, y_train, lambda_reg=0.01)
    acc_LAND, mse_LAND, selected_feature_names_LAND = classify(X_train, X_test, y_train, y_test, selected_features_LAND, classifier)
    print("LAND selected features:")
    print_selected_features(acc_LAND, mse_LAND, selected_features_LAND, selected_feature_names_LAND)
    print("\n\n")

    selected_features_hsic_lasso = fs.hsiclasso(X_train, y_train, num_feat=20)
    acc_hsic_lasso, mse_hsic_lasso, selected_feature_names_hsic_lasso = classify(X_train, X_test, y_train, y_test, selected_features_hsic_lasso, classifier)
    print("HSIC Lasso selected features:")
    print_selected_features(acc_hsic_lasso, mse_hsic_lasso, selected_features_hsic_lasso, selected_feature_names_hsic_lasso)
    print("\n\n")

    selected_features_mRMR = fs.mRMR(X_train, y_train, num_features_to_select=50)
    acc_mRMR, mse_mRMR, selected_feature_names_mRMR = classify(X_train, X_test, y_train, y_test, selected_features_mRMR, classifier)
    print("mRMR selected features:")
    print_selected_features(acc_mRMR, mse_mRMR, selected_features_mRMR, selected_feature_names_mRMR)
    print("\n\n")

    selected_features_gumbel = fs.select_features_gumbel(X_train, y_train, num_features_to_select=50)
    acc_gumbel, mse_gumbel, selected_feature_names_gumbel = classify(X_train, X_test, y_train, y_test, selected_features_gumbel, classifier)
    print("Gumbel selected features:")
    print_selected_features(acc_gumbel, mse_gumbel, selected_features_gumbel, selected_feature_names_gumbel)
    print("\n\n")

    selected_features_permutation = fs.Perm_importance(X_train, y_train, classifier=classifier, select_features=X_clustered)
    acc_permutation, mse_permutation, selected_feature_names_permutation = classify(X_train, X_test, y_train, y_test, selected_features_permutation, classifier)
    print("Permutation Importance selected features clustered:")
    print_selected_features(acc_permutation, mse_permutation, selected_features_permutation, selected_feature_names_permutation)
    print("\n\n")

    selected_features_permutation_mRMR = fs.Perm_importance(X_train, y_train, classifier=classifier, select_features=X_mRMR)
    acc_permutation_mRMR, mse_permutation_mRMR, selected_feature_names_permutation_mRMR = classify(X_train, X_test, y_train, y_test, selected_features_permutation_mRMR, classifier)
    print("Permutation Importance selected features mRMR:")
    print_selected_features(acc_permutation_mRMR, mse_permutation_mRMR, selected_features_permutation_mRMR, selected_feature_names_permutation_mRMR)
    print("\n\n")

    selected_features_sfs = fs.backwards_SFS(X_train, y_train, 20, classifier, X_clustered)
    acc_sfs, mse_sfs, selected_feature_names_sfs = classify(X_train, X_test, y_train, y_test, selected_features_sfs, classifier)
    print("Sequential Feature Selection (SFS) selected features:")
    print_selected_features(acc_sfs, mse_sfs, selected_features_sfs, selected_feature_names_sfs)
    print("\n\n")

    selected_features_sfs_mRMR = fs.backwards_SFS(X_train, y_train, 20, classifier, X_mRMR)
    acc_sfs_mRMR, mse_sfs_mRMR, selected_feature_names_sfs_mRMR = classify(X_train, X_test, y_train, y_test, selected_features_sfs_mRMR, classifier)
    print("Sequential Feature Selection (SFS) selected features mRMR:")
    print_selected_features(acc_sfs_mRMR, mse_sfs_mRMR, selected_features_sfs_mRMR, selected_feature_names_sfs_mRMR)
    print("\n\n")

if __name__ == '__main__':
    main()
