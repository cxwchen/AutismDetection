#%%
import numpy as np
from math import sqrt, log
from generator import graph_generator, signal_generator
from algorithms import lADMM
from sklearn.covariance import empirical_covariance
import matplotlib.pyplot as plt
from algorithms import *
from generator import *

# Function to compute F-measure between two adjacency matrices
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

def threshold_and_normalize_adjacency(adj_matrix, threshold=0):
    """
    Normalizes the weights of the adjacency matrix to be between 0 and 1 using min-max normalization.
    
    Parameters:
    - adj_matrix: A numpy 2D array representing the adjacency matrix.
    
    Returns:
    - normalized_adj: The adjacency matrix with normalized weights between 0 and 1.
    """
    # Find the minimum and maximum values in the adjacency matrix
    min_val = np.min(adj_matrix)
    max_val = np.max(adj_matrix)
    
    # Apply min-max normalization
    normalized_adj = (adj_matrix - min_val) / (max_val - min_val)  # Scales the values between 0 and 1
    
    # Apply absolute value and then thresholding
    normalized_adj = np.where(np.abs(normalized_adj) >= threshold, normalized_adj, 0)    
    return normalized_adj

def sample_graph(num_samples=10000, n=20, p=0.2):
    # Generate ER graph adjacency matrix
    
    signals = np.zeros((num_samples, n))

    # Generate multiple filtered signals from different random inputs
    for i in range(num_samples):
        x_i = np.random.randn(n, 1)
        filtered_signal, _ = signal_generator(W, x_i, n, f='quadratic_filter')
        signals[i, :] = filtered_signal.flatten()

    # Estimate covariance matrix empirically
    cov_est = empirical_covariance(signals)

    # Initialize lADMM variables
    m = n
    s0 = np.ones((m, m))
    q0 = s0.dot(np.ones((m, 1)))
    Z0 = s0.copy()
    lambda20 = np.zeros((m, m))
    lambda30 = np.zeros((m, 1))

    alpha = 0.5
    rho = 1
    tau1 = 0.7
    epsilon = 1e-6
    kMax = 10000
    delta = 10 * sqrt(log(num_samples) / num_samples)

    # Run lADMM
    s, ss, r, rrho = lADMM(cov_est, s0, Z0, q0, lambda20, lambda30, alpha, delta, rho, tau1, epsilon, kMax)

    # Normalize s between 0 and 1 (if necessary, for example)
    s_normalized = (s - np.min(s)) / (np.max(s) - np.min(s))

    # Threshold to binarize s (set values above 0.5 to 1, others to 0)
    s_binarized = (s_normalized > 0.2642).astype(int)

    # Compute F-measure between original adjacency W and learned s
    fscore = f_measure(W, s_binarized)
    print(f"F-measure between original W and learned s: {fscore:.4f}")

    # Plot adjacency matrices side by side

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(W, cmap='viridis')
    plt.title("True Adjacency Matrix")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(s_binarized, cmap='viridis')
    plt.title("Learned Adjacency Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    #return s_binarized

def multiple_graph_avg_F(kMax):
    num_graphs = 10

    # Store the F-measure scores
    f_measures = []
    optimal_thresholds = []
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

        # Initialize lADMM variables
        m = n
        s0 = np.ones((m, m))
        q0 = s0.dot(np.ones((m, 1)))
        Z0 = s0.copy()
        lambda20 = np.zeros((m, m))
        lambda30 = np.zeros((m, 1))

        alpha = 0.5
        rho = 1.0
        tau1 = 0.8
        epsilon = 1e-6
        #kMax = 2000
        delta = 20 * sqrt(log(num_samples) / num_samples)
        
        # Run lADMM
        s, ss, r, rrho = lADMM(cov_est, s0, Z0, q0, lambda20, lambda30, alpha, delta, rho, tau1, epsilon, kMax)

        # Normalize s between 0 and 1 (if necessary)
        s_normalized = (s - np.min(s)) / (np.max(s) - np.min(s))

        # Try different thresholds and compute F-measure for each
        thresholds = np.linspace(0, 1, 20)  # Test thresholds from 0 to 1 in 20 steps
        best_fscore = -np.inf
        best_threshold = 0

        for threshold in thresholds:
            s_binarized = (s_normalized > threshold).astype(int)
            fscore = f_measure(W, s_binarized)
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
    
def multiple_graph():
    num_graphs = 5

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

        # Initialize lADMM variables
        m = n
        s0 = np.ones((m, m))
        q0 = s0.dot(np.ones((m, 1)))
        Z0 = s0.copy()
        lambda20 = np.zeros((m, m))
        lambda30 = np.zeros((m, 1))

        alpha = 0.5
        rho = 1.0
        tau1 = 0.80
        epsilon = 1e-6
        kMax = 2000
        delta = 0.001 * sqrt(log(num_samples) / num_samples)
        
        # Run lADMM
        s, ss, r, rrho = lADMM(cov_est, s0, Z0, q0, lambda20, lambda30, alpha, delta, rho, tau1, epsilon, kMax)

        # Normalize s between 0 and 1 (if necessary)
        s_normalized = (s - np.min(s)) / (np.max(s) - np.min(s))
        
        s_binarized = (s_normalized > 0.1).astype(int)

        # Compute F-measure between original adjacency W and learned s
        fscore = f_measure(W, s_binarized)
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

def multiple_k():
    num_graphs = 30

    # Range for kMax (from 1000 to 10000)
    kmax_values = range(1000, 10001, 1000)

    # Store the F-measure and optimal threshold for each kMax value
    f_measures_per_kmax = {}
    optimal_thresholds_per_kmax = {}

    # Loop over 10 random graphs and different values of kMax
    for kMax in kmax_values:
        f_measures = []
        optimal_thresholds = []
        
        # Loop over 10 random graphs
        for graph_idx in range(num_graphs):
            # Generate random ER graph adjacency matrix
            W = graph_generator(N=n, p=p, m0=0, m=0, type='ER')

            signals = np.zeros((num_samples, n))

            # Generate multiple filtered signals from different random inputs
            for i in range(num_samples):
                x_i = np.random.randn(n, 1)  # subgaussian input: Uniform[-1, 1]
                filtered_signal, _ = signal_generator(W, x_i, n, f='lowpass_exp_filter')
                signals[i, :] = filtered_signal.flatten()

            # Estimate covariance matrix empirically
            cov_est = empirical_covariance(signals)

            # Initialize lADMM variables
            m = n
            s0 = np.ones((m, m))
            q0 = s0.dot(np.ones((m, 1)))
            Z0 = s0.copy()
            lambda20 = np.zeros((m, m))
            lambda30 = np.zeros((m, 1))

            alpha = 0.5
            rho = 1.0
            tau1 = 0.7
            epsilon = 1e-6
            delta = 20 * sqrt(log(num_samples) / num_samples)
            
            # Run lADMM
            s, ss, r, rrho = lADMM(cov_est, s0, Z0, q0, lambda20, lambda30, alpha, delta, rho, tau1, epsilon, kMax)

            # Normalize s between 0 and 1 (if necessary)
            s_normalized = (s - np.min(s)) / (np.max(s) - np.min(s))

            # Try different thresholds and compute F-measure for each
            thresholds = np.linspace(0, 1, 20)  # Test thresholds from 0 to 1 in 20 steps
            best_fscore = -np.inf
            best_threshold = 0

            for threshold in thresholds:
                s_binarized = (s_normalized > threshold).astype(int)
                fscore = f_measure(W, s_binarized)
                if fscore > best_fscore:
                    best_fscore = fscore
                    best_threshold = threshold

            # Store the best threshold and F-measure for this graph
            f_measures.append(best_fscore)
            optimal_thresholds.append(best_threshold)

        # Compute the average F-measure and optimal threshold for this kMax
        avg_f_measure = np.mean(f_measures)
        avg_optimal_threshold = np.mean(optimal_thresholds)

        # Store the results for this kMax
        f_measures_per_kmax[kMax] = avg_f_measure
        optimal_thresholds_per_kmax[kMax] = avg_optimal_threshold

    # Print the results
    print("F-measures and optimal thresholds for each kMax:")
    for kMax in kmax_values:
        print(f"kMax = {kMax}: Avg F-measure = {f_measures_per_kmax[kMax]:.4f}, Avg Optimal Threshold = {optimal_thresholds_per_kmax[kMax]:.4f}")

    # Optionally, plot F-measures and optimal thresholds for different kMax values
    plt.figure(figsize=(10, 6))
    plt.plot(kmax_values, list(f_measures_per_kmax.values()), label='Avg F-measure', marker='o')
    plt.plot(kmax_values, list(optimal_thresholds_per_kmax.values()), label='Avg Optimal Threshold', marker='s')
    plt.xlabel("kMax")
    plt.ylabel("Value")
    plt.title("F-measures and Optimal Thresholds vs kMax")
    plt.legend()
    plt.grid(True)
    plt.show()

n=20
p=0.2
W = graph_generator(n,p,m0=n,m=n/2,type='ER')
sample_graph()
#plot_ladmm_relative_error_multiple_samples()

# %%
