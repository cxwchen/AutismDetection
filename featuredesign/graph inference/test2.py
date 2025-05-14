#%%
import numpy as np
import matplotlib.pyplot as plt

def create_ring_graph(N):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i - 1) % N] = 1
        A[i, (i + 1) % N] = 1
    return A

def compute_normalized_laplacian(A):
    d = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-10)))
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return L

def compute_polynomial_filter(L, alpha=0.5, K=5):
    H = np.zeros_like(L)
    Lk = np.eye(L.shape[0])
    for k in range(K + 1):
        H += (alpha ** k) * Lk
        Lk = Lk @ L
    return H

def covariance_convergence_plot():
    N = 10
    alpha = 0.5
    K = 5
    sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    errors = []

    # Graph and filter
    A = create_ring_graph(N)
    L = compute_normalized_laplacian(A)
    H = compute_polynomial_filter(L, alpha=alpha, K=K)
    C_true = H @ H.T

    for P in sample_sizes:
        Z = np.random.randn(N, P)
        X = H @ Z
        X_centered = X - np.mean(X, axis=1, keepdims=True)
        C_sample = (1 / (P - 1)) * X_centered @ X_centered.T

        error = np.linalg.norm(C_sample - C_true, ord='fro')
        errors.append(error)
        print(f"P={P:<5}  Frobenius error: {error:.6f}")
    print(C_sample)
    print(C_true)
    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(sample_sizes, errors, marker='o')
    plt.xscale('log')
    plt.xlabel("Number of Samples (log scale)")
    plt.ylabel("Frobenius Norm Error ||Ĉ_x - C_x||")
    plt.title("Convergence of Sample Covariance to True Covariance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run it
covariance_convergence_plot()
