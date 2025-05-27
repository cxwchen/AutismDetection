import numpy as np
import cvxpy as cp
from numpy.linalg import eigh, norm
from math import sqrt, log
import matplotlib.pyplot as plt
from tqdm import trange, tqdm  # Add this at the top with other imports
from algorithms import *
from generator import *
from sklearn.covariance import empirical_covariance

# --------- Laplacian Learning ---------
def normalized_laplacian(V_hat, epsilon=1e-1, alpha=0.5):
    """
    Learns a normalized Laplacian S such that:
    - S is PSD, symmetric, with 1s on diagonal
    - Off-diagonal entries in [-1, 0]
    - S approximates spectral form S' = ∑ λ_k v_k v_kᵀ
    - S' has smallest eigenvalue (λ₁) = 0, enforced via λ₀ = 0
    """
    N = V_hat.shape[0]
    S = cp.Variable((N, N), symmetric=True)
    lambda_vec = cp.Variable(N)

    # S' = V Λ Vᵀ
    S_prime = sum([lambda_vec[k] * np.outer(V_hat[:, k], V_hat[:, k]) for k in range(N)])

    I = np.eye(N)
    off_diag_mask = np.ones((N, N)) - I
    off_diag_entries = cp.multiply(off_diag_mask, S)

    constraints = [
        S >> 0,                                   # PSD
        cp.diag(S) == 1,                          # Diagonal = 1
        #off_diag_entries <= 0,                    # Off-diagonal ≤ 0
        off_diag_entries >= -1,                   # Off-diagonal ≥ -1
        cp.norm(S - S_prime, 'fro') <= epsilon,   # Spectral similarity
        lambda_vec[0] == 0,                       # λ₁(S') = 0, right eigenvector picked?
    ]

    # Objective: sparsity in S
    objective = cp.Minimize(alpha * cp.norm(S, 1))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    # print("Step 2 Optimization Status:", problem.status)
    return S.value

def normalized_laplacian_reweighted(V_hat, tau=1.0, delta=1e-4, epsilon=0.1, max_iter=10):
    
    N = V_hat.shape[0]
    S_est = np.zeros((N, N))
    weights = np.ones((N, N))
    off_diag_mask = np.ones((N, N)) - np.eye(N)  # to exclude diagonal

    I = np.eye(N)
    off_diag_mask = np.ones((N, N)) - I
    
    for p in range(max_iter):
        S = cp.Variable((N, N))
        lambda_vec = cp.Variable(N)
        S_prime = sum([lambda_vec[k] * np.outer(V_hat[:, k], V_hat[:, k]) for k in range(N)])
        off_diag_entries = cp.multiply(off_diag_mask, S)
        
        constraints = [
            cp.diag(S) == 1,                          # Diagonal = 1
            S >> 0,                                   # PSD
            off_diag_entries >= -1,                   # Off-diagonal ≥ -1
            cp.norm(S - S_prime, 'fro') <= epsilon,   # Spectral similarity
            lambda_vec[0] == 0                       # λ₁(S') = 0, right eigenvector picked?  
        ]

        objective = cp.Minimize(cp.sum(cp.multiply(weights * off_diag_mask, cp.abs(S))))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if S.value is None:
            print(f"Iteration {p+1}: optimization failed.")
            break

        S_est = S.value
        weights = tau / (np.abs(S_est) + delta)
        weights *= off_diag_mask  # keep ignoring the diagonal
        nnz = (np.abs(S_est) > 1e-3).sum()
        print(f"Iter {p+1:>2}: status = {problem.status}, nonzeros = {nnz}")

    return S_est

# --------- Adjacency Learning ---------
def adjacency_reweighted(V_hat, tau=1.0, delta=1e-5, epsilon=0.1, max_iter=10, binarize_threshold=0.1):
    
    N = V_hat.shape[0]
    S_est = np.zeros((N, N))
    weights = np.ones((N, N))
    off_diag_mask = np.ones((N, N)) - np.eye(N)

    for p in trange(max_iter, desc="Learning adjacency matrix", unit="iter"):
        S = cp.Variable((N, N))
        lambda_vec = cp.Variable(N)
        S_prime = sum([lambda_vec[k] * np.outer(V_hat[:, k], V_hat[:, k]) for k in range(N)])

        t1 = cp.Variable()
        t2 = cp.Variable()

        constraints = [
            cp.diag(S) == 0,
            S == S.T,
            t1 >= cp.norm(S - S_prime, 'fro'),
            t2 >= cp.abs(cp.sum(S[:, 0]) - 1),
            cp.norm(cp.hstack([t1, t2]), 2) <= epsilon    
        ]

        objective = cp.Minimize(cp.sum(cp.multiply(weights * off_diag_mask, cp.abs(S))))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if S.value is None:
            tqdm.write(f"Iteration {p+1}: optimization failed.")
            break

        S_est = S.value
        weights = tau / (np.abs(S_est) + delta)
        weights *= off_diag_mask
        nnz = (np.abs(S_est) > 1e-3).sum()
        
    return S_est

def learn_adjacency_rLogSpecT(V_hat, delta_n, alpha=0.1):
    """
    Learn adjacency matrix using rLogSpecT formulation without reweighting
    
    min_S ||S||_{1,1} - α * 1^T log(S1)
    s.t. ||S C_n - C_n S||_F^2 ≤ δ_n^2
         S = S^T, diag(S) = 0, S ≥ 0
    
    Parameters:
        X: Signal matrix (N x P)
        alpha: Weight for log-barrier term
        delta_n: Commutator constraint threshold
    """
    N = V_hat.shape[0]
    
    S = cp.Variable((N, N), nonneg=True)
    
    # Objective terms
    l1_term = cp.sum(cp.abs(S))  # ||S||_{1,1}
    log_term = cp.sum(cp.log(cp.sum(S, axis=1) + 1e-10))  # 1^T log(S1)
    
    # Constraints
    constraints = [
        S == S.T,
        cp.diag(S) == 0,
        cp.sum_squares(S @ V_hat - V_hat @ S) <= delta_n**2
    ]
    
    # Solve problem
    problem = cp.Problem(cp.Minimize(l1_term - alpha * log_term), constraints)
    problem.solve(solver=cp.SCS)
    
    if S.value is None:
        raise ValueError("Optimization failed to converge")
        
    return S.value

def binarize_adjacency_matrix(adj_matrix, threshold=0.1, keep_diagonal=False):
    """
    Binarizes an adjacency matrix based on a threshold value.
    
    Parameters:
    -----------
    adj_matrix : numpy.ndarray
        Input adjacency matrix (can be weighted)
    threshold : float, optional (default=0.1)
        Values above this threshold become 1, others become 0
    keep_diagonal : bool, optional (default=False)
        Whether to preserve the diagonal values or force them to 0
    
    Returns:
    --------
    numpy.ndarray
        Binarized adjacency matrix with 0/1 values
    """
    # Create a copy to avoid modifying the original matrix
    binary_matrix = adj_matrix.copy()
    
    # Apply threshold
    binary_matrix = (binary_matrix > threshold).astype(float)
    
    # Zero out diagonal unless specified to keep it
    if not keep_diagonal:
        np.fill_diagonal(binary_matrix, 0)
    
    return binary_matrix