#%%
import numpy as np
import cvxpy as cp
from numpy.linalg import eigh, norm
import matplotlib.pyplot as plt
from pygsp import *
from sklearn.covariance import empirical_covariance
from featuredesign.graph_inference.GSP_methods import *

# --------- Graph Generators ---------
def graph_generator(N,p,m0=0,m=0,type='ER'):
    if type == 'ER':
        G = graphs.ErdosRenyi(N=N, p=p, directed=0, self_loops=0)
    elif type == 'BA':
        G = graphs.BarabasiAlbert(N=N, m0=m0, m=m)
    elif type == 'SBM':
        G = graphs.StochasticBlockModel(N = N,k = 2, p = 0.7, q = 0.1)
    return G.W.toarray()
    # G.W is the weighted matrix represented by array matrix
    
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

def sample_covariance(X):
    """
    Compute the sample covariance matrix for data X.
    Each column of X is a sample, each row is a variable.
    Returns a (n_variables, n_variables) covariance matrix.
    """
    X = np.asarray(X)
    mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean
    n_samples = X.shape[1]
    cov = (X_centered @ X_centered.T) / (n_samples - 1)
    return cov

def main():
    N = 20
    p=0.2
    P_values = np.array([100, 200, 500, 1000, 2000, 5000, 10000])
    
    filter = [1, 1, 1]
    
    frob_errors = []
    rel_errors = []
    
    # Generate the true normalized Laplacian
    L_true = graph_generator(N, p, type='ER')
    
    # Find the smallest epsilon for which L_learned is valid (not None)
    for P in P_values:
        X = simulate_diffused_graph_signals(L_true, filter=filter, P=int(P))
        L_learned = learn_adjacency_LADMM(empirical_covariance(X), 20 * np.sqrt(np.log(P) / P))  # find_min_epsilon(X, epsilon_values)
        mask_offdiag = np.ones_like(L_true) - np.eye(L_true.shape[0])
        L_true_masked = L_true #* mask_offdiag
        L_learned_masked = L_learned #* mask_offdiag
        
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
    N = 20
    p = 0.2
    P_values = np.array([100, 200, 500, 1000, 2000, 5000, 10000])
    
    filter = [1, 1, 1]
    
    frob_errors = []
    rel_errors = []
    
    # Generate the true normalized Laplacian
    L_true = graph_generator(N, p, type='ER')
    
    # Find the smallest epsilon for which L_learned is valid (not None)
    for P in P_values:
        X = simulate_diffused_graph_signals(L_true, filter=filter, P=int(P))
        L_learned = learn_adjacency_LADMM(sample_covariance(X), 7 * np.sqrt(np.log(P) / P))  # find_min_epsilon(X, epsilon_values)
        # L_learned = (L_learned > 0.2642).astype(int)
        # Compute Frobenius and relative errors
        frob_error = norm(L_true - L_learned, 'fro')
        rel_error = frob_error / norm(L_true, 'fro')
        
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
# %%
