#%%
import numpy as np
import cvxpy as cp
from numpy.linalg import eigh, norm
import matplotlib.pyplot as plt
from pygsp import *

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

def graph_generator(N,p,m0=0,m=0,type='ER'):
    if type == 'ER':
        G = graphs.ErdosRenyi(N=N, p=p, directed=0, self_loops=0)
    elif type == 'BA':
        G = graphs.BarabasiAlbert(N=N, m0=m0, m=m)
    elif type == 'SBM':
        G = graphs.StochasticBlockModel(N = N,k = 2, p = 0.7, q = 0.1)
    return G.W.toarray()
    # G.W is the weighted matrix represented by array matrix
    
def create_random_normalized_laplacian(N, mean=0.0, std=1.0):
    # Step 1: Create random symmetric adjacency with positive weights
    A_upper = np.random.normal(mean, std, size=(N, N))
    A_upper = np.triu(np.abs(A_upper), k=1)  # abs to ensure positive weights
    
    A = A_upper + A_upper.T
    np.fill_diagonal(A, 0)
    
    # Step 2: Compute degree matrix
    degrees = A.sum(axis=1)
    # To avoid division by zero for isolated nodes:
    degrees[degrees == 0] = 1e-10
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    
    # Step 3: Compute normalized Laplacian
    L_norm = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    
    return L_norm

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
def simulate_diffused_graph_signals(S, filter, P=100):
    N = S.shape[0]
    signals = np.zeros((N, P))
    for i in range(P):
        z = np.random.randn(N)
        x = np.zeros(N)
        Sk = np.eye(N)
        for term in filter:
            x += term * Sk @ z
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
def refine_normalized_laplacian_with_spectrum(V_hat, epsilon=1e-1, alpha=0.5):
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

    print("Optimization Status:", problem.status)
    return S.value, problem.status

def learn_normalized_laplacian(X, epsilon=1e-1, alpha=0.5):
    sample_cov, _ = compute_sample_covariance(X)
    eigvals, V_hat = eigh(sample_cov)
    return refine_normalized_laplacian_with_spectrum(V_hat, epsilon, alpha)

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

    # Mask out diagonal
    mask_offdiag = np.ones_like(A_true) - np.eye(A_true.shape[0])
    A_true_masked = A_true * mask_offdiag
    A_learned_masked = A_learned * mask_offdiag

    frob_error = norm(A_true_masked - A_learned_masked, 'fro')
    rel_error = frob_error / (norm(A_true_masked, 'fro') + 1e-10)  # avoid division by zero
    shd = np.sum(np.abs(A_true_bin - A_learned_bin))

    print("\n=== Graph Comparison ===")
    print(f"Frobenius Error:        {frob_error:.4f}")
    print(f"Relative Error:         {rel_error:.4f}")

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

def threshold_and_normalize_laplacian(adj_matrix, threshold=0):
    
    return np.where(np.abs(adj_matrix) >= threshold, adj_matrix, 0)

def find_min_epsilon(simulated_signals, epsilon_values):
    # Placeholder to store the smallest valid epsilon and corresponding L_learned
    L_learned_valid = None
    
    for epsilon in epsilon_values:
        # Assuming learn_normalized_laplacian function returns None if it fails
        L_learned, status = learn_normalized_laplacian(X=simulated_signals, epsilon=epsilon, alpha=0.5)

        if status == 'infeasible':
            print('limit reached')
            break  # Exit once a valid L_learned is found
    
        L_learned_valid = L_learned
    return threshold_and_normalize_laplacian(L_learned_valid, 0.05)

def SampleLoop():
    N = 20
    P_values = np.array([100, 200, 500, 1000, 2000, 5000, 10000])
    epsilon_values = np.linspace(0.01, 0.3, 20)[::-1]  # Test epsilon values from 0.01 to 1
    p = 0.2
    filter = [1, 1, 1]
    
    frob_errors = []
    rel_errors = []
    
    # Generate the true normalized Laplacian
    L_true = compute_normalized_laplacian(graph_generator(N, p, type='ER'))
    
    # Find the smallest epsilon for which L_learned is valid (not None)
    for P in P_values:
        X = simulate_diffused_graph_signals(L_true, filter=filter, P=int(P))
        L_learned = find_min_epsilon(X, epsilon_values)      
        mask_offdiag = np.ones_like(L_true) - np.eye(L_true.shape[0])
        L_true_masked = L_true * mask_offdiag
        L_learned_masked = L_learned * mask_offdiag
        
        # Compute Frobenius and relative errors
        frob_error = norm(L_true_masked - L_learned_masked, 'fro')
        rel_error = frob_error / norm(L_true_masked, 'fro')
        
        # Store the errors
        frob_errors.append(frob_error)
        rel_errors.append(rel_error)

    #compare_graphs(L_true, L_learned)
    plot_graphs(L_true, L_learned)
    plt.figure(figsize=(8, 5))
    plt.plot(P_values, rel_errors, label="Relative Error", marker='s', color='red')
    plt.xscale('log')
    plt.xlabel("Number of Samples P")
    plt.ylabel("Error")
    plt.title("Convergence of Learned Laplacian to True Laplacian")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# --------- Main Pipeline ---------
if __name__ == "__main__":
    #N = 20
    N_values = [10,20,30,40,50]
    P = 10000
    #P_values = np.array([100, 200, 500, 1000, 2000, 5000, 10000])
    epsilon_values = np.linspace(0.01, 0.3, 20)[::-1]  # Test epsilon values from 0.01 to 1
    p = 0.2
    filter = [1, 1, 1]
    
    frob_errors = []
    rel_errors = []
    
    # Generate the true normalized Laplacian
    
    
    # Find the smallest epsilon for which L_learned is valid (not None)
    for N in N_values:
        L_true = compute_normalized_laplacian(graph_generator(N, p, type='ER'))
        X = simulate_diffused_graph_signals(L_true, filter=filter, P=int(P))
        L_learned = find_min_epsilon(X, epsilon_values)      
        mask_offdiag = np.ones_like(L_true) - np.eye(L_true.shape[0])
        L_true_masked = L_true * mask_offdiag
        L_learned_masked = L_learned * mask_offdiag
        
        # Compute Frobenius and relative errors
        frob_error = norm(L_true_masked - L_learned_masked, 'fro')
        rel_error = frob_error / norm(L_true_masked, 'fro')
        
        # Store the errors
        frob_errors.append(frob_error)
        rel_errors.append(rel_error)

    #compare_graphs(L_true, L_learned)
    plot_graphs(L_true, L_learned)
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, rel_errors, label="Relative Error", marker='s', color='red')
    plt.xlabel("Number of nodes N")
    plt.ylabel("Error")
    plt.title("Relative error of ER graph with N nodes after 10^5 samples")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# %%
