import numpy as np
import cvxpy as cp
from numpy.linalg import eigh, norm
import matplotlib.pyplot as plt

# --------- Covariance ---------
def compute_sample_covariance(X):
    P = X.shape[1]
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    sample_cov = (1 / (P - 1)) * X_centered @ X_centered.T
    return sample_cov, X_centered

# --------- Laplacian Learning ---------
def refine_normalized_laplacian_with_spectrum(V_hat, epsilon=1e-1, alpha=0.01):
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

def learn_normalized_laplacian(X, epsilon=1e-1, alpha=0.1):
    # print("Step 1: Covariance and Eigendecomposition")
    sample_cov, _ = compute_sample_covariance(X)
    eigvals, V_hat = eigh(sample_cov)
    # print(V_hat)
    print("Step 2: Learning Laplacian")
    return refine_normalized_laplacian_with_spectrum(V_hat, epsilon, alpha)

