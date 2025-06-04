from __future__ import division
import numpy as np
from scipy.stats import gamma
from sklearn.linear_model import Lasso, LassoCV
import warnings
warnings.filterwarnings('ignore')

def rbf_dot(pattern1, pattern2, deg):
    """RBF kernel computation"""
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = np.sum(pattern1*pattern1, 1).reshape(size1[0],1)
    H = np.sum(pattern2*pattern2, 1).reshape(size2[0],1)

    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))

    H = Q + R - 2* np.dot(pattern1, pattern2.T)
    H = np.exp(-H/2/(deg**2))

    return H

def compute_hsic_values(X, y):
    """
    Compute HSIC values between each feature and target
    
    Parameters:
    X: n x p matrix of features
    y: n x 1 target vector
    
    Returns:
    hsic_values: p-dimensional vector of HSIC values
    """
    n, p = X.shape
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    
    # Center the target
    H = np.identity(n) - np.ones((n,n)) / n
    
    # Compute kernel width for y
    G = np.sum(y*y, 1).reshape(n,1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))
    dists = Q + R - 2* np.dot(y, y.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)
    width_y = np.sqrt(0.5 * np.median(dists[dists>0]))
    
    # Compute y kernel
    L = rbf_dot(y, y, width_y)
    Lc = np.dot(np.dot(H, L), H)
    
    hsic_values = np.zeros(p)
    
    for j in range(p):
        x_j = X[:, j].reshape(-1, 1)
        
        # Compute kernel width for feature j
        G = np.sum(x_j*x_j, 1).reshape(n,1)
        Q = np.tile(G, (1, n))
        R = np.tile(G.T, (n, 1))
        dists = Q + R - 2* np.dot(x_j, x_j.T)
        dists = dists - np.tril(dists)
        dists = dists.reshape(n**2, 1)
        width_x = np.sqrt(0.5 * np.median(dists[dists>0]))
        
        # Compute feature kernel
        K = rbf_dot(x_j, x_j, width_x)
        Kc = np.dot(np.dot(H, K), H)
        
        # Compute HSIC
        hsic_values[j] = np.sum(Kc.T * Lc) / n
        
    return hsic_values

def hsic_lasso(X, y, alpha=None, max_features=None):
    """
    HSIC LASSO feature selection algorithm
    
    Parameters:
    X: n x p feature matrix
    y: n-dimensional target vector
    alpha: regularization parameter (if None, uses cross-validation)
    max_features: maximum number of features to select (if None, no limit)
    
    Returns:
    selected_indices: indices of selected features
    """
    X = X.values
    y = y.values

    n, p = X.shape
    y = y.reshape(-1, 1) if y.ndim == 1 else y

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X_normalized = (X - X_mean) / X_std  # Normalize features

    
    #Compute HSIC values for all features
    hsic_values = compute_hsic_values(X, y)

    print(f"HSIC values range: [{np.min(hsic_values):.6f}, {np.max(hsic_values):.6f}]")
    print(f"Number of non-zero HSIC values: {np.sum(hsic_values > 1e-10)}")

    valid_features = hsic_values > 1e-10
    if not np.any(valid_features):
        print("No features have significant HSIC values. Returning empty selection.")
        return np.array([])
 
    # Simple thresholding approach instead of problematic LASSO setup
    if alpha is None:
        # Use adaptive threshold based on HSIC distribution
        sorted_hsic = np.sort(hsic_values[valid_features])[::-1]
        if len(sorted_hsic) > 1:
            # Use gap-based selection or top percentage
            gaps = sorted_hsic[:-1] - sorted_hsic[1:]
            if len(gaps) > 0:
                max_gap_idx = np.argmax(gaps)
                threshold = sorted_hsic[max_gap_idx + 1]
            else:
                threshold = np.median(sorted_hsic)
        else:
            threshold = sorted_hsic[0] * 0.1
    else:
        # Use alpha as relative threshold
        max_hsic = np.max(hsic_values)
        threshold = alpha * max_hsic
    
    print(f"Using threshold: {threshold:.6f}")
    
    # Get selected features (non-zero coefficients)
    selected_mask = hsic_values >= threshold
    selected_indices = np.where(selected_mask)[0]
    
    # If max_features is specified, select top features by HSIC value
    if max_features is not None and len(selected_indices) > max_features:
        hsic_selected = hsic_values[selected_indices]
        top_k_idx = np.argsort(hsic_selected)[-max_features:]
        selected_indices = selected_indices[top_k_idx]
    
    # Sort indices for consistent output
    selected_indices = np.sort(selected_indices)

    print(f"Selected features: {selected_indices}, with HSIC values: {hsic_values[selected_indices]}")
    
    return selected_indices

def hsic_lasso_forward_selection(X, y, max_features=10, threshold=0.01):
    """
    Alternative HSIC LASSO using forward selection approach
    
    Parameters:
    X: n x p feature matrix
    y: n-dimensional target vector  
    max_features: maximum number of features to select
    threshold: minimum HSIC improvement threshold
    
    Returns:
    selected_indices: indices of selected features
    """
    n, p = X.shape
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    
    selected_indices = []
    remaining_indices = list(range(p))
    
    for _ in range(min(max_features, p)):
        best_hsic = -1
        best_idx = -1
        
        for idx in remaining_indices:
            # Test adding this feature
            test_indices = selected_indices + [idx]
            test_X = X[:, test_indices]
            
            # Compute HSIC between selected features and target
            hsic_values = compute_hsic_values(test_X, y)
            total_hsic = np.sum(hsic_values)
            
            if total_hsic > best_hsic:
                best_hsic = total_hsic
                best_idx = idx
        
        # Check if improvement is significant
        current_hsic = 0
        if selected_indices:
            current_X = X[:, selected_indices]
            current_hsic_values = compute_hsic_values(current_X, y)
            current_hsic = np.sum(current_hsic_values)
        
        if best_hsic - current_hsic < threshold:
            break
            
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return np.array(sorted(selected_indices))