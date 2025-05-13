from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from skfeature import classifiers as cl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def hsic(X, Y, gamma=1.0):
    # Compute the kernel matrix for X and Y using the RBF kernel
    Kx = rbf_kernel(np.array(X).reshape(-1, 1), gamma=gamma)
    Ky = rbf_kernel(np.array(Y).reshape(-1, 1), gamma=gamma)
    
    # Centering the kernel matrices
    n = Kx.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    
    # Compute the HSIC
    hsic_value = np.trace(Kx @ H @ Ky.T @ H) / (n - 1)**2
    return hsic_value

def hsic_lasso(X, y, alpha=1.0, max_iter=100):
   # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute the HSIC values for each feature
    hsic_values = []
    for i in range(X_scaled.shape[1]):
        hsic_value = hsic(X_scaled[:, i], y)
        hsic_values.append(hsic_value)
    
    hsic_values = np.array(hsic_values)
    
    # Perform Lasso regression to select features based on HSIC values
    lasso = LassoLars(alpha=alpha, max_iter=max_iter)
    lasso.fit(X_scaled, y)
    
    # Select the features with non-zero coefficients
    selected_features = np.where(lasso.coef_ != 0)[0]
    
    return selected_features 

def train_and_evaluate(X, y, selected_features, test_size=0.2,random_state=40):
    #splitting the data in train and test 0.8:0.2 respecively
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    #changing the test and train data to the data selected by the feature selection tool
    selected_train_x = X_train.iloc[:, selected_features]
    selected_test_X = X_test.iloc[:, selected_features]

    #scaling the features, otherwise the maximum iterations of the classifier will be exceded
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(selected_train_x)
    X_test_scaled = scaler.transform(selected_test_X)

    #applying the classifier from the classifier group
    model = cl.applySVM(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    #finding mse and accuracy
    mse = mean_squared_error(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    #getting and printing the feature names
    feature_names = X_train.columns
    selected_feature_names = feature_names[selected_features]
    
    return acc, mse, selected_feature_names

def cross_validate_model(X, y, selected_features, n_splits=5):
    """K-Fold cross-validation evaluation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores = []
    mse_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train_sel = X_train.iloc[:, selected_features]
        X_test_sel = X_test.iloc[:, selected_features]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

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
    # Load with encoding fix
    data = pd.read_csv('ABIDEII_Composite_Phenotypic.csv', encoding='ISO-8859-1')

    # Drop rows where DX_GROUP is missing
    data = data[data['DX_GROUP'].notna()]

    # Separate X and y
    X = data.drop(columns=['DX_GROUP'])
    y = data['DX_GROUP']

    # Encode categorical
    X = pd.get_dummies(X, drop_first=True)

    #drop the columns where all values are missing before the imputer
    X = X.dropna(axis=1, how='all')

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

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
