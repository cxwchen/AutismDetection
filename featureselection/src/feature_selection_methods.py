from __future__ import division
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoLars
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import RFE, SequentialFeatureSelector, VarianceThreshold, mutual_info_classif, mutual_info_regression
from skfeature.function.information_theoretical_based import MRMR
from scipy.stats import gamma
from pyHSICLasso import HSICLasso
from torch import nn
import torch
import torch.nn.functional as F
import warnings

class GumbelFeatureSelector(nn.Module):
    def __init__(self, in_features, out_features, tau=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.logits = nn.Parameter(torch.randn(in_features, out_features))

    def sample_gumbel_softmax(self, logits, eps=1e-20):
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y = logits + gumbel_noise
        return F.softmax(y / self.tau, dim=0)

    def forward(self, x, hard=False):
        weights = self.sample_gumbel_softmax(self.logits)
        if hard:
            index = weights.max(dim=0, keepdim=True)[1]
            one_hot = torch.zeros_like(weights).scatter_(0, index, 1.0)
            weights = (one_hot - weights).detach() + weights
        return torch.matmul(x, weights)

class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_classes):
        super().__init__()
        self.conv1 = nn.Linear(in_features, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, out_classes)

    def forward(self, x, adj):
        x = torch.spmm(adj, x)
        x = F.relu(self.conv1(x))
        x = torch.spmm(adj, x)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)
    
def select_features_gumbel(X, y, num_features_to_select, tau=1.0, epochs=100, lr=0.01):
    """
    Selects features using a Gumbel-Softmax based feature selector.
    
    Parameters:
    - X: Input feature matrix (numpy array or torch tensor).
    - y: Target labels (numpy array or torch tensor).
    - num_features_to_select: Number of features to select.
    - tau: Temperature parameter for Gumbel-Softmax.
    - epochs: Number of training epochs.
    
    Returns:
    - Selected feature indices.
    """

    X_values = X.values
    y_values = y.values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_tensor = torch.tensor(X_values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_values, dtype=torch.long).to(device)
    
    model = GumbelFeatureSelector(in_features=X_tensor.shape[1], out_features=num_features_to_select, tau=tau).to(device)
    
    classifier = nn.Linear(num_features_to_select, len(torch.unique(y_tensor))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        classifier.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        logits = classifier(output)
        loss = F.cross_entropy(output, y_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        selected_weights = model(X_tensor, hard=True)
    selected_indices = torch.nonzero(selected_weights.sum(dim=1) > 0).flatten().cpu().numpy()

    return selected_indices.tolist()

def rbf_dot(pattern1, pattern2, deg):
	size1 = pattern1.shape
	size2 = pattern2.shape

	G = np.sum(pattern1*pattern1, 1).reshape(size1[0],1)
	H = np.sum(pattern2*pattern2, 1).reshape(size2[0],1)

	Q = np.tile(G, (1, size2[0]))
	R = np.tile(H.T, (size1[0], 1))

	H = Q + R - 2* np.dot(pattern1, pattern2.T)

	H = np.exp(-H/2/(deg**2))

	return H

def hsic_gam(X, Y, alph = 0.5):
    """
	X, Y are numpy vectors with row - sample, col - dim
	alph is the significance level
	auto choose median to be the kernel width
	"""
    n = X.shape[0]
    valid_dists = []
	# ----- width of X -----
    Xmed = X

    G = np.sum(Xmed*Xmed, 1).reshape(n,1)
    Q = np.tile(G, (1, n) )
    R = np.tile(G.T, (n, 1) )

    dists = Q + R - 2* np.dot(Xmed, Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)
    valid_dists = dists[dists>0]

    if len(valid_dists) == 0:
        width_x = 1.0
    else:
        width_x = np.sqrt(0.5 * np.median(valid_dists))
                
    if width_x <= 0 or np.isnan(width_x):
        width_x = 1.0

    return (width_x)

def hsiclasso(X, y, num_feat=20):
    """
    Perform HSIC Lasso feature selection.
    Parameters:
    - X: Input feature matrix (numpy array or pandas DataFrame).
    - y: Target labels (numpy array or pandas Series).
    - alpha: Regularization strength.
    - max_iter: Maximum number of iterations for convergence.
    - tol: Tolerance for convergence.
    Returns:
    - Selected feature indices.
    """
    
    # Ensure X and y are numpy arrays with proper data types
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    # Ensure y is EXACTLY 1D (pyHSICLasso is very strict about this)
    if len(y.shape) > 1:
        y = y.flatten()
    
    print(f"Final shapes - X: {X.shape}, y: {y.shape}")
    # Perform HSIC Lasso to select features
    hsic_lasso = HSICLasso()
    # Set parameters for HSIC Lasso

    # Fit the model
    hsic_lasso.input(X, y)
    hsic_lasso.classification(num_feat)

    selected_features = hsic_lasso.get_features()

         # Convert string indices to integers if necessary
    if len(selected_features) > 0 and isinstance(selected_features[0], str):
        try:
            selected_features = [int(feat) for feat in selected_features]
            print(f"Converted to integer indices: {selected_features}")
        except ValueError as e:
            print(f"Could not convert feature names to integers: {e}")
            # If conversion fails, try to map to column positions
            if hasattr(X, 'columns'):
                # If X is a DataFrame, map feature names to positions
                feature_positions = []
                for feat in selected_features:
                    try:
                        pos = list(X.columns).index(feat)
                        feature_positions.append(pos)
                    except ValueError:
                        print(f"Feature {feat} not found in columns")
                selected_features = feature_positions
                print(f"Mapped to column positions: {selected_features}")

    return selected_features

def select_model(classifier):
    # Determine the model based on the classifier name
    if classifier == "SVM":
        model = SVC(kernel='linear')
    elif classifier == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    elif classifier == "LogR":
        model = LogisticRegression(random_state=42)
    elif classifier == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42)
    elif classifier == "MLP":
        model = MLPClassifier(random_state=42)
    else:
        raise ValueError("Unsupported classifier type")
    
    return model

def low_variance(X, threshold=0.5):
    print("Total features before low variance filter: ", X.shape[1])
    model = VarianceThreshold(threshold=threshold)
    X_reduced = model.fit_transform(X)
    selected_features = model.get_support(indices=True)
    print("Total features after low variance filter: ", X_reduced.shape[1])
    return X_reduced, selected_features

def greedy_hsic_lasso(X, y, k, redundancy_penalty=0.5):
   # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_reshaped = np.array(y).reshape(-1,1)

    n_features = X.shape[1]
    selected = []
    remaining = list(range(n_features))

    #Compute the HSIC values relevance
    relevance = []
    for i in range(n_features):
        X_feat = X_scaled[:,i].reshape(-1,1)
        hsic_value, _ = hsic_gam(X_feat, y_reshaped, alph=0.5)
        relevance.append(hsic_value)
    relevance = np.array(relevance)

    #Greedy feature selection
    for _ in range(k):
        best_score = -np.inf
        best_feature = None

        for i in remaining:
            redundancy = 0
            for j in selected:
                non_score, _ = hsic_gam(X_scaled[:,i].reshape(-1,1), X_scaled[:,j].reshape(-1,1), alph=0.5)
                redundancy += non_score
            score = relevance[i] - redundancy_penalty * redundancy

            if score > best_score:
                best_score = score
                best_feature = i

        selected.append(best_feature)
        remaining.remove(best_feature)
        print(f"Selected feature {best_feature} with score {best_score:.4f}")

    return selected

def mRMR(X, y, num_features_to_select=50):

    X_array = X.values
    y_array = np.asarray(y).ravel()  # Ensure y is a 1D array
    X_array = X_array.astype(np.float64)  # Ensure X is float64 for compatibility

        # Handle categorical target variable
    if y_array.dtype == 'object' or not np.issubdtype(y_array.dtype, np.number):
        le = LabelEncoder()
        y_array = le.fit_transform(y_array)
    
    y_array = y_array.astype(np.int32)  # MRMR often expects integer labels

        # Check for NaN values and handle them
    if np.any(np.isnan(X_array)) or np.any(np.isnan(y_array)):
        print("Warning: NaN values detected. Consider handling them before feature selection.")
        #Remove rows with NaN
        valid_rows = ~(np.isnan(X_array).any(axis=1) | np.isnan(y_array))
        X_array = X_array[valid_rows]
        y_array = y_array[valid_rows]

    mRMR_selector = MRMR.mrmr(X_array, y_array)

    selected_indices = mRMR_selector[0:num_features_to_select]

    return selected_indices

def LAND(X, y, lambda_reg=0.01):

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    n_samples, n_features = X.shape

    width_y = hsic_gam(y, y, alph=0.5)
    H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    L = rbf_dot(y, y, width_y)
    Lc = H @ L @ H
    L_vec = Lc.ravel()

    if np.allclose(L_vec, 0, atol=1e-12):
        warnings.warn("Target kernel is degenerate (all zeros)")
        return np.array([], dtype=int)
    
    K_vecs = np.zeros((n_samples**2, n_features))
    valid_features = []

    for k in range(n_features):
        feature_k = X[:, k].reshape(-1, 1)

        if np.var(feature_k) == 0:
            continue  # Skip constant features

        width_x = hsic_gam(feature_k, feature_k, alph=0.5)
        K = rbf_dot(feature_k, feature_k, width_x)
        Kc = H @ K @ H
        K_vec = Kc.ravel()

        if not np.allclose(K_vec, 0, atol=1e-12):
            K_vecs[:, len(valid_features)] = K_vec
            valid_features.append(k)
    
    K_vecs = K_vecs[:, :len(valid_features)]  # Keep only valid features

    effective_lambda = max(lambda_reg, 1e-12)  # Ensure lambda is not too small

    model = Lasso(alpha=effective_lambda, fit_intercept=False)
    model.fit(K_vecs, L_vec)

    score = model.coef_

    selected_mask = np.abs(score) > 1e-10  # Threshold to filter out near-zero coefficients
    selected_features = np.array(valid_features)[selected_mask]

    return selected_features

def lars_lasso(X, y, alpha=0.1, max_iter=100):
   # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_reshaped = np.array(y).reshape(-1,1)
        
    # Perform Lasso regression to select features based on HSIC values
    lasso = LassoLars(alpha=alpha, max_iter=max_iter, eps=1e-6)
    lasso.fit(X_scaled, y_reshaped)
    
    # Select the features with non-zero coefficients
    selected_features = np.where(lasso.coef_ != 0)[0]
    
    return selected_features 

def Perm_importance(X, y, classifier, min_features=30, select_features=None):

    # Determine the model based on the classifier name
    model = select_model(classifier)

    if select_features is not None:
        if isinstance(select_features, list):
            select_features = [X.columns.get_loc(col) for col in select_features]  # Convert column names to indices
        X = X.iloc[:, select_features]
    
    model.fit(X, y)

    # Calculate permutation importance
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)

    # Get the importances and sort them from most to least important
    importances = result.importances_mean
    indices = np.argsort(importances)[::-1]
    
    # Select features based on importance (threshold: features that have positive importance)
    selected_features = [i for i in indices if importances[i] > 0]  # Select features that have positive importance

        # Fallback: ensure at least `min_features` are returned
    if len(selected_features) < min_features:
        selected_features = indices[:min_features].tolist()

    return selected_features

def RecursiveFE(X, y, n_features_to_select, classifier, select_features=None):
    # Determine the model based on the classifier name
    model = select_model(classifier)

    if select_features is not None:
        # Select the features based on the selected indices 
        X = X.loc[:, select_features]

    # Initialize RFE with the base model and the desired number of features to select
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)

    # Fit RFE
    rfe.fit(X, y)

    # Get the selected feature indices
    selected_features = np.where(rfe.support_)[0]

    return selected_features

def backwards_SFS(X, y, n_features_to_select, classifier, select_features=None):
    # Determine the model based on the classifier name
    model = select_model(classifier)

    if select_features is not None:
        if isinstance(select_features, list):
            select_features = [X.columns.get_loc(col) for col in select_features]  # Convert column names to indices
        X = X.iloc[:, select_features]

    # Initialize SequentialFeatureSelector with the base model and the desired number of features to select
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select, direction='backward')

    # Fit SFS
    sfs.fit(X, y)

    # Get the selected feature indices
    selected_features = np.where(sfs.get_support())[0]

    return selected_features

def l1_logistic_regression(X, y, C=1, max_iter=100):
    """
    Perform L1 regularized logistic regression to select features.
    
    Parameters:
    - X: Input feature matrix (numpy array or pandas DataFrame).
    - y: Target labels (numpy array or pandas Series).
    - alpha: Regularization strength.
    - max_iter: Maximum number of iterations for convergence.
    
    Returns:
    - Selected feature indices.
    """
    
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit L1 logistic regression model
    model = LogisticRegression(penalty='l1', C=C, solver='liblinear', max_iter=max_iter)
    model.fit(X_scaled, y)
    
    # Get the selected feature indices
    selected_features = np.where(model.coef_[0] != 0)[0]
    
    return selected_features
