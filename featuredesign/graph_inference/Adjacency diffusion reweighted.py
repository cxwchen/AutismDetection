#%%
import numpy as np
import cvxpy as cp
from numpy.linalg import eigh, norm
import matplotlib.pyplot as plt
from tqdm import trange, tqdm  # Add this at the top with other imports

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

def create_random_normal_adjacency(N, mean=0.0, std=1.0):
    # Generate upper triangle random normal entries
    A_upper = np.random.normal(mean, std, size=(N, N))
    A_upper = np.triu(A_upper, k=1)

    # Symmetrize to get full adjacency
    A = A_upper + A_upper.T

    # Zero diagonal (no self loops)
    np.fill_diagonal(A, 0)

    # Normalize only the first column
    col_sum = np.sum(A[:, 0])
    if col_sum != 0:
        A[:, 0] /= col_sum
        A[0, :] = A[:, 0]  # Maintain symmetry on first row as well
    return A

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

# --------- Adjacency Learning ---------
def learn_adjacency_matrix(X, tau=1.0, delta=1e-5, epsilon=0.1, max_iter=10, binarize_threshold=0.1):
    sample_cov, _ = compute_sample_covariance(X)
    _, V_hat = eigh(sample_cov)
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
        #tqdm.write(f"Iter {p+1:>2}: status = {problem.status}, nonzeros = {nnz}")

    return S_est


# --------- Evaluation ---------
def compare_graphs(A_true, A_learned, threshold=1e-3):
    A_true_bin = (A_true > threshold).astype(int)
    A_learned_bin = (A_learned > threshold).astype(int)

    tp = np.sum((A_true_bin == 1) & (A_learned_bin == 1))
    fp = np.sum((A_true_bin == 0) & (A_learned_bin == 1))
    fn = np.sum((A_true_bin == 1) & (A_learned_bin == 0))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    frob_error = norm(A_true - A_learned, 'fro')
    rel_error = frob_error / norm(A_true, 'fro')
    shd = np.sum(np.abs(A_true_bin - A_learned_bin))

    print("\n=== Graph Comparison ===")
    print(f"Frobenius Error:        {frob_error:.4f}")
    print(f"Relative Error:         {rel_error:.4f}")
    print(f"Structural Hamming Distance (SHD): {shd}")
    print(f"Precision:              {precision:.4f}")
    print(f"Recall:                 {recall:.4f}")
    print(f"F1 Score:               {f1:.4f}")

def plot_graphs(S_true, S_learned):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(S_true, cmap='viridis')
    plt.title("True Adjacency Matrix")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(S_learned, cmap='viridis')
    plt.title("Learned Adjacency Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# --------- Main Pipeline ---------
if __name__ == "__main__":
    #np.random.seed(0)
    N = 40
    P = 250
    graph_type = "community"  # Choose from: "ring", "star", "community"
    
    A_true = create_random_normal_adjacency(N, mean=0.0, std=1.0)
    
    #A_true = select_graph(graph_type, N)
    X = simulate_diffused_graph_signals(A_true, P=P)
    A_learned = learn_adjacency_matrix(X, epsilon=0.05, max_iter=100)
    compare_graphs(A_true, A_learned)
    plot_graphs(A_true, A_learned)

# %%
