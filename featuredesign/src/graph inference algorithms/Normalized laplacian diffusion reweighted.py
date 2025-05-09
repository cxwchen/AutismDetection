# %%
import numpy as np
import cvxpy as cp
from numpy.linalg import eigh, norm
import matplotlib.pyplot as plt

# --------- Graph Generators ---------
def create_ring_graph(N):
    W = np.zeros((N, N))
    for i in range(N):
        W[i, (i - 1) % N] = 1
        W[i, (i + 1) % N] = 1
    return W

def create_star_graph(N):
    W = np.zeros((N, N))
    for i in range(1, N):
        W[0, i] = 1
        W[i, 0] = 1
    return W

def create_community_graph(N1, N2, inter_prob=0.05):
    N = N1 + N2
    W = np.zeros((N, N))
    for i in range(N1):
        for j in range(i + 1, N1):
            W[i, j] = W[j, i] = 1
    for i in range(N1, N):
        for j in range(i + 1, N):
            W[i, j] = W[j, i] = 1
    for _ in range(int(inter_prob * N1 * N2)):
        i = np.random.randint(0, N1)
        j = np.random.randint(N1, N)
        W[i, j] = W[j, i] = 1
    return W

def select_graph(graph_type, N):
    if graph_type == "ring":
        return create_ring_graph(N)
    elif graph_type == "star":
        return create_star_graph(N)
    elif graph_type == "community":
        return create_community_graph(N // 2, N - N // 2)
    else:
        raise ValueError("Invalid graph type")

# --------- Laplacian Computation ---------
def compute_normalized_laplacian(W):
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D.diagonal(), 1e-10)))
    return np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

# --------- Signal Simulation ---------
def simulate_diffused_graph_signals(S, P=100, K=5, alpha=0.5):
    N = S.shape[0]
    signals = np.zeros((N, P))
    for i in range(P):
        z = np.random.randn(N)
        x = np.zeros(N)
        Sk = np.eye(N)
        for k in range(K + 1):
            x += (alpha ** k) * Sk @ z
            Sk = Sk @ S
        signals[:, i] = x
    return signals

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
        off_diag_entries <= 0,                    # Off-diagonal ≤ 0
        off_diag_entries >= -1,                   # Off-diagonal ≥ -1
        cp.norm(S - S_prime, 'fro') <= epsilon,   # Spectral similarity
        lambda_vec[0] == 0                        # λ₁(S') = 0, right eigenvector picked?
    ]

    # Objective: sparsity in S
    objective = cp.Minimize(alpha * cp.norm(S, 1))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    print("Step 2 Optimization Status:", problem.status)
    return S.value



def learn_normalized_laplacian(X, epsilon=1e-1, alpha=0.1):
    print("Step 1: Covariance and Eigendecomposition")
    sample_cov, _ = compute_sample_covariance(X)
    eigvals, V_hat = eigh(sample_cov)
    print(V_hat)
    print("Step 2: Learning Laplacian")
    return refine_normalized_laplacian_with_spectrum(V_hat, epsilon, alpha)

# --------- Evaluation ---------
def compare_graphs(S_true, S_learned):
    frob_error = norm(S_true - S_learned, 'fro')
    rel_error = frob_error / norm(S_true, 'fro')
    threshold = 1e-3
    A_true = (S_true < -threshold).astype(int)
    A_learned = (S_learned < -threshold).astype(int)
    shd = np.sum(np.abs(A_true - A_learned))

    print("\n=== Graph Comparison ===")
    print(f"Frobenius Error: {frob_error:.4f}")
    print(f"Relative Error:  {rel_error:.4f}")
    print(f"SHD:             {shd}")

def plot_graphs(S_true, S_learned):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(S_true, cmap='viridis')
    plt.title("True Normalized Laplacian")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(S_learned, cmap='viridis')
    plt.title("Learned Normalized Laplacian")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# --------- Main Pipeline ---------
if __name__ == "__main__":
    np.random.seed(0)
    N = 10
    P = 1000
    graph_type = "community"  # Choose: "ring", "star", "community"

    W = select_graph(graph_type, N)
    L_true = compute_normalized_laplacian(W)
    X = simulate_diffused_graph_signals(L_true, P=P)
    L_learned = learn_normalized_laplacian(X, epsilon=0.1, alpha=0.01)

    compare_graphs(L_true, L_learned)
    plot_graphs(L_true, L_learned)

# %%
