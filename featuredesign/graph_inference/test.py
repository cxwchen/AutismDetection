#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# --------- Graph & Laplacian ---------
def create_ring_graph(N):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i - 1) % N] = 1
        A[i, (i + 1) % N] = 1
    return A

def compute_normalized_laplacian(A):
    degrees = np.sum(A, axis=1)
    print(np.sqrt(degrees))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
    return np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

# --------- Diffusion Filter ---------
def compute_polynomial_filter(S, alpha=0.5, K=5):
    N = S.shape[0]
    H = np.zeros((N, N))
    Sk = np.eye(N)
    for k in range(K + 1):
        H += (alpha ** k) * Sk
        Sk = Sk @ S
    return H

# --------- Cosine similarity with best alignment ---------
def aligned_cosine_similarity(V1, V2):
    cos_sim = np.abs(V1.T @ V2)
    cost = 1 - cos_sim
    row_ind, col_ind = linear_sum_assignment(cost)
    aligned = np.abs(V1.T @ V2[:, col_ind])
    return np.mean(np.diag(aligned))  # average alignment

# --------- Convergence Experiment ---------
def eigenvector_convergence_plot(N=10, K=50, alpha=0.5):
    A = create_ring_graph(N)
    S = compute_normalized_laplacian(A)
    H = compute_polynomial_filter(S, alpha=alpha, K=K)
    eigvals_S, V_S = np.linalg.eigh(S)

    sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000]
    similarities = []

    for P in sample_sizes:
        Z = np.random.randn(N, P)
        X = H @ Z
        X_centered = X - np.mean(X, axis=1, keepdims=True)
        C_x = (1 / (P - 1)) * X_centered @ X_centered.T
        _, V_Cx = np.linalg.eigh(C_x)

        score = aligned_cosine_similarity(V_S, V_Cx)
        similarities.append(score)
        print(f"P={P:<5} → average aligned eigenvector cosine similarity: {score:.4f}")
    print(V_Cx)
    # Plot convergence
    plt.figure(figsize=(7, 4))
    plt.plot(sample_sizes, similarities, marker='o')
    plt.xlabel("Number of Samples (P)")
    plt.ylabel("Avg Cosine Similarity with S eigenvectors")
    plt.title("Convergence of C_x Eigenvectors to Laplacian Eigenvectors")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run the experiment
eigenvector_convergence_plot(N=5)


# %%
