#%%
import numpy as np
import cvxpy as cp
from numpy.linalg import eigh, norm
from math import sqrt, log
import matplotlib.pyplot as plt
from tqdm import trange, tqdm  # Add this at the top with other imports
from algorithms import *
from generator import *
from sklearn.covariance import empirical_covariance

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

# --------- Adjacency Learning ---------
def learn_adjacency_rLogSpecT(sample_cov, delta_n, alpha=0.1):
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
    N = sample_cov.shape[0]
    
    S = cp.Variable((N, N), nonneg=True)
    
    # Objective terms
    l1_term = cp.sum(cp.abs(S))  # ||S||_{1,1}
    log_term = cp.sum(cp.log(cp.sum(S, axis=1) + 1e-10))  # 1^T log(S1)
    
    # Constraints
    constraints = [
        S == S.T,
        cp.diag(S) == 0,
        cp.sum_squares(S @ sample_cov - sample_cov @ S) <= delta_n**2
    ]
    
    # Solve problem
    problem = cp.Problem(cp.Minimize(l1_term - alpha * log_term), constraints)
    problem.solve(solver=cp.SCS)
    
    if S.value is None:
        raise ValueError("Optimization failed to converge")
        
    return S.value


# --------- Evaluation ---------
def f_measure(W_true, W_pred, threshold=1e-4):
    # Binarize matrices: edges are values > threshold
    W_true_bin = (W_true > threshold).astype(int)
    W_pred_bin = (W_pred > threshold).astype(int)
    
    # Flatten matrices
    true_edges = W_true_bin.flatten()
    pred_edges = W_pred_bin.flatten()
    
    # True Positives (TP): edges correctly predicted
    TP = np.sum((true_edges == 1) & (pred_edges == 1))
    # False Positives (FP): predicted edges that don't exist
    FP = np.sum((true_edges == 0) & (pred_edges == 1))
    # False Negatives (FN): edges missed by prediction
    FN = np.sum((true_edges == 1) & (pred_edges == 0))
    
    # Precision and Recall
    precision = TP / (TP + FP + 1e-10)  # add small eps to avoid div zero
    recall = TP / (TP + FN + 1e-10)
    
    # F-measure (harmonic mean)
    if precision + recall == 0:
        return 0.0
    fscore = 2 * precision * recall / (precision + recall)
    return fscore

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

def multiple_graph(n,p,num_samples):
    num_graphs = 1

    # Store the F-measure scores
    f_measures = []
    # Loop over 10 random graphs
    for graph_idx in range(num_graphs):
        # Generate random ER graph adjacency matrix
        W = graph_generator(N=n, p=p, m0=0, m=0, type='ER')

        signals = np.zeros((num_samples, n))

        # Generate multiple filtered signals from different random inputs
        for i in range(num_samples):
            x_i = np.random.randn(n, 1)  # subgaussian input: Uniform[-1, 1]
            filtered_signal, _ = signal_generator(W, x_i, n, f='quadratic_filter')
            signals[i, :] = filtered_signal.flatten()

        # Estimate covariance matrix empirically
        cov_est = empirical_covariance(signals)

        # Initialize rLogSpecT variables
        delta_n = 5 * sqrt(log(num_samples) / num_samples)
        alpha = 0.5
        
        # Run rLogSpecT
        A_learned = learn_adjacency_rLogSpecT(cov_est,delta_n,alpha)
        
        # Normalize s between 0 and 1 (if necessary)
        A_normalized = (A_learned - np.min(A_learned)) / (np.max(A_learned) - np.min(A_learned))
        
        A_binarized = (A_normalized > 0.2647).astype(int)

        # Compute F-measure between original adjacency W and learned s
        fscore = f_measure(W, A_binarized)
        f_measures.append(fscore)


    # Calculate the average F-measure
    avg_f_measure = np.mean(f_measures)

    print(f"Average F-measure over {num_graphs} random graphs: {avg_f_measure:.4f}")
    # Optionally, you can plot the distribution of F-measures and optimal thresholds
    plt.figure(figsize=(8, 6))
    plt.hist(f_measures, bins=10, alpha=0.7, color='blue', label="F-measures")
    plt.axvline(avg_f_measure, color='red', linestyle='dashed', linewidth=2, label=f"Avg F-measure: {avg_f_measure:.4f}")
    plt.title("Distribution of F-measures over Random Graphs")
    plt.xlabel("F-measure")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def multiple_graph_avg_F(n,p,num_samples):
    num_graphs = 100

    # Store the F-measure scores
    f_measures = []
    optimal_thresholds = []
    # Loop over 10 random graphs
    for graph_idx in range(num_graphs):
        
        W = graph_generator(N=n, p=p, m0=0, m=0, type='ER')

        signals = np.zeros((num_samples, n))

        # Generate multiple filtered signals from different random inputs
        for i in range(num_samples):
            x_i = np.random.randn(n, 1)  # subgaussian input: Uniform[-1, 1]
            filtered_signal, _ = signal_generator(W, x_i, n, f='quadratic_filter')
            signals[i, :] = filtered_signal.flatten()

        # Estimate covariance matrix empirically
        cov_est = empirical_covariance(signals)

        # Initialize rLogSpecT variables
        delta_n = 20 * sqrt(log(num_samples) / num_samples)
        alpha = 0.5
        
        # Run rLogSpecT
        A_learned = learn_adjacency_rLogSpecT(cov_est,delta_n,alpha)
        
        # Normalize s between 0 and 1 (if necessary)
        A_normalized = (A_learned - np.min(A_learned)) / (np.max(A_learned) - np.min(A_learned))

        # Try different thresholds and compute F-measure for each
        thresholds = np.linspace(0, 1, 20)  # Test thresholds from 0 to 1 in 20 steps
        best_fscore = -np.inf
        best_threshold = 0

        for threshold in thresholds:
            A_binarized = (A_normalized > threshold).astype(int)
            fscore = f_measure(W, A_binarized)
            if fscore > best_fscore:
                best_fscore = fscore
                best_threshold = threshold

        # Store the best threshold and F-measure
        f_measures.append(best_fscore)
        optimal_thresholds.append(best_threshold)

    # Calculate the average F-measure
    avg_f_measure = np.mean(f_measures)
    avg_optimal_threshold = np.mean(optimal_thresholds)

    print(f"Average F-measure over {num_graphs} random graphs: {avg_f_measure:.4f}")
    print(f"Average optimal threshold: {avg_optimal_threshold:.4f}")
    # Optionally, you can plot the distribution of F-measures and optimal thresholds
    plt.figure(figsize=(8, 6))
    plt.hist(f_measures, bins=10, alpha=0.7, color='blue', label="F-measures")
    plt.axvline(avg_f_measure, color='red', linestyle='dashed', linewidth=2, label=f"Avg F-measure: {avg_f_measure:.4f}")
    plt.title("Distribution of F-measures over Random Graphs")
    plt.xlabel("F-measure")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(optimal_thresholds, bins=10, alpha=0.7, color='green', label="Optimal Thresholds")
    plt.axvline(avg_optimal_threshold, color='red', linestyle='dashed', linewidth=2, label=f"Avg Optimal Threshold: {avg_optimal_threshold:.4f}")
    plt.title("Distribution of Optimal Thresholds over Random Graphs")
    plt.xlabel("Threshold Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
# --------- Main Pipeline ---------
if __name__ == "__main__":
    N = 20
    P = 250
    multiple_graph_avg_F(N,0.4,P)