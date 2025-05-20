from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.metrics.pairwise import rbf_kernel
from HSIC import hsic_gam
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from skfeature import classifiers as cl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from skfeature.AAL_test import multiset_feats
from skfeature.AAL_test import load_files
from skfeature.AAL_test import multiset_pheno

def load_file():
    folder_path = r"C:\Users\guus\Python_map\AutismDetection-main\abide\female-cpac-filtnoglobal-aal" # Enter your local ABIDE dataset path
    data_arrays, file_paths, subject_ids, metadata = load_files(folder_path)

    feature_df = multiset_feats(data_arrays, file_paths, subject_ids)
    print("Extracted features shape:\n", feature_df.shape)

    full_df = multiset_pheno(feature_df)
    print("Merged feature+label shape:\n", full_df.shape)

    print("subject ids: ", subject_ids)

    X = full_df.drop(columns=['DX_GROUP', 'subject_id'])
    y = full_df['DX_GROUP']

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

    # Encode categorical
    X = pd.get_dummies(X, drop_first=True)

    #drop the columns where all values are missing before the imputer
    X = X.dropna(axis=1, how='all')

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return(X, y)

def hsic(X, y, sigma_x=1.0, sigma_y=1.0):
    
    # Compute the kernel matrix for X and Y using the RBF kernel
    K = rbf_kernel(np.array(X).reshape(-1, 1), gamma=1.0 / (2 * sigma_x **2))
    L = rbf_kernel(np.array(y).reshape(-1, 1), gamma=1.0 / (2 * sigma_y ** 2))

    # Centering the kernel matrices
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    
    # Compute the HSIC
    hsic_value = np.trace(K @ H @ L.T @ H) / (n - 1)**2

    #K_centered = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
    #L_centered = L - L.mean(axis=0, keepdims=True) - L.mean(axis=1, keepdims=True) + L.mean()

    #norm_K = np.linalg.norm(K_centered, 'fro')
    #norm_L = np.linalg.norm(L_centered, 'fro')

    #hsic_value = np.trace(np.dot(K_centered, L_centered)) / (norm_K * norm_L)
    return hsic_value

def hsic_lasso(X, y, alpha=1.0, max_iter=100):
   # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_reshaped = np.array(y).reshape(-1,1)
    
    # Compute the HSIC values for each feature
    hsic_values = []
    for i in range(X_scaled.shape[1]):
        X_feat = X_scaled[:,i].reshape(-1,1)
        hsic_value, _ = hsic_gam(X_feat, y_reshaped, alph=0.5)
        hsic_values.append(hsic_value)
    
    hsic_values = np.array(hsic_values)
    
    # Perform Lasso regression to select features based on HSIC values
    lasso = LassoLars(alpha=alpha, max_iter=max_iter, eps=1e-6)
    lasso.fit(X_scaled, y)
    
    # Select the features with non-zero coefficients
    selected_features = np.where(lasso.coef_ != 0)[0]
    
    return selected_features 

def train_and_evaluate(X, y, selected_features):
    #splitting the data in train and test 0.8:0.2 respecively
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    print(f"Selected features ({len(selected_features)}):", selected_features)
    #changing the test and train data to the data selected by the feature selection tool
    selected_train_x = X_train.iloc[:, selected_features]
    selected_test_X = X_test.iloc[:, selected_features]

    #scaling the features, otherwise the maximum iterations of the classifier will be exceded
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(selected_train_x)
    X_test_scaled = scaler.transform(selected_test_X)

    #applying the classifier from the classifier group
    model = cl.applyLogR(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    #finding mse and accuracy
    mse = mean_squared_error(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    #getting and printing the feature names
    feature_names = X.columns
    selected_feature_names = feature_names[selected_features]
    
    return acc, mse, selected_feature_names

def cross_validate_model(X, y, selected_features, n_splits=5):
    #K-Fold cross-validation evaluation.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores = []
    mse_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Select the features based on the selected indices 
        X_train_sel = X_train.iloc[:, selected_features]
        X_test_sel = X_test.iloc[:, selected_features]

        #Scaling the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        #applying the classifier
        model = cl.applyLogR(X_train_scaled, y_train)
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

def main():

    X, y = load_file()

    #performing lasso feature selection
    selected_features = hsic_lasso(X, y, alpha=0.1)
    print(f"Selected features ({len(selected_features)}):", selected_features)

    acc, mse, selected_feature_names = train_and_evaluate(X, y, selected_features)
    #acc, mse, selected_feature_names = cross_validate_model(X, y, selected_features)
    print(f"Train/Test Accuracy: {acc:.4f}, MSE: {mse:.4f}")
    print(f"\nSelected feature names({len(selected_feature_names)}):")
    for name in selected_feature_names:
        print("-", name)

if __name__ == '__main__':
    main()
