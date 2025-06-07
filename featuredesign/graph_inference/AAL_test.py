#%%
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os,glob,re,random, contextlib, io
import pandas as pd
import seaborn as sns
import networkx.algorithms.community as nx_comm
import cvxpy as cp
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import pearsonr,skew, kurtosis, entropy
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV, LedoitWolf, empirical_covariance
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations
from scipy.signal import find_peaks
from sklearn.decomposition import FastICA,PCA
from scipy.linalg import pinv, eigh, norm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from nilearn import datasets, image
from nilearn.input_data import NiftiLabelsMasker
from typing import List, Union
from networkx.linalg.laplacianmatrix import laplacian_matrix
from sklearn.metrics import adjusted_rand_score

from featuredesign.graph_inference.GSP_methods import *

import warnings
warnings.filterwarnings("once", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
_issued_warnings = set()

def check_stationarity(fmri_data, alpha=0.05):
    nonstationary_counts = []
    for subj in fmri_data:
        count = 0
        for roi in range(subj.shape[1]):
            pval = adfuller(subj[:, roi])[1]
            if pval > alpha:
                count += 1
        nonstationary_counts.append(count)
    return nonstationary_counts

def zeroVar_filter(fmri_data, subject_ids, variance_threshold=1e-6):
    """
    Remove subjects with zero-variance ROIs while maintaining ID consistency

    Args:
        fmri_data: List of subject time series arrays
        subject_ids: List of corresponding subject IDs
        variance_threshold: Minimum acceptable standard deviation

    Returns:
        clean_data: Filtered time series data
        clean_ids: Corresponding subject IDs
        bad_indices: Original indices of removed subjects
    """
    clean_data = []
    clean_ids = []
    bad_indices = []

    for idx, (data, sid) in enumerate(zip(fmri_data, subject_ids)):
        if data is None:
            bad_indices.append(idx)
            continue

        roi_stds = np.std(data, axis=0)
        if np.any(roi_stds < variance_threshold):
            bad_indices.append(idx)
        else:
            clean_data.append(data)
            clean_ids.append(sid)

    print(f"Removed {len(bad_indices)}/{len(fmri_data)} subjects")
    return clean_data, clean_ids, bad_indices

def pk_extract(x, time_values=None, height_threshold=0, prominence=1):
    """
    Extract and Compute statistics of peaks (if peaks exist).

    Returns:
            'num_peaks': int,
            'mean_amplitude': float,
            'max_amplitude': float,
            'mean_time_interval': float (if time_values provided)
    """
    peaks, properties = find_peaks(x, height=height_threshold, prominence=prominence)

    peaks_data = {
        'peak_amplitudes': x[peaks],
        'peak_indices': peaks
    }

    if time_values is not None:
        peaks_data['peak_times'] = time_values[peaks]

    stats = {
        'num_peaks': len(peaks_data['peak_amplitudes']),
        'mean_amplitude': np.mean(peaks_data['peak_amplitudes']) if peaks_data['peak_amplitudes'].size > 0 else 0,
        'max_amplitude': np.max(peaks_data['peak_amplitudes']) if peaks_data['peak_amplitudes'].size > 0 else 0
    }

    if 'peak_times' in peaks_data and len(peaks_data['peak_times']) > 1:
        time_intervals = np.diff(peaks_data['peak_times'])
        stats['mean_time_interval'] = np.mean(time_intervals)

    return stats, peaks_data

def pk_stats(list_of_timeseries, time_values=None):
    """
Takes list of timeseries and returns peak features per timeseries as a dataframe

Features: avg peak interval, max peak, avg peak amplitude
    """
    all_stats = []
    for i, ts in enumerate(list_of_timeseries):
        stats, peaks = pk_extract(ts, time_values)
        #stats['ROI'] = i+1  # Track which series these stats belong to
        all_stats.append(stats)

    return pd.DataFrame(all_stats)

def stat_feats(x, include_stat_features=True, include_corr_matrix=True):
    """
    Compute classic statistical features based on some time series data and stores it to a panda dataframe.

    Features:
    - Mean, Standard Deviation, Skewness, Kurtosis, Slope, Correlation, Covariance, Signal-to-noise ratio etc.
    """
    # Ensure shape is (n_rois, n_timepoints)
    if x.shape[0] > x.shape[1]:
        x = x.T

    n_rois = x.shape[0]
    output_parts = []

    if include_stat_features:
        feature_list = []
        for ts in x:
            features = {
                'mean': np.mean(ts),
                'std': np.std(ts),
                'SNR': np.mean(ts) / (np.std(ts) + 1e-10),
                'Skewness': skew(ts),
                'Kurtosis': kurtosis(ts)
            }
            feature_list.append(features)

        df_stats = pd.DataFrame(feature_list)
        df_stats.insert(0, 'ROI', [f"ROI_{i + 1}" for i in range(n_rois)])
        output_parts.append(df_stats)

    if include_corr_matrix:
        corr_matrix = np.corrcoef(x)
        corr_features = {
            f"corr_ROI_{i+1}_ROI_{j+1}": corr_matrix[i, j]
            for i, j in combinations(range(n_rois), 2)
        }
        df_corr = pd.DataFrame([corr_features])  # one row for all pairwise correlations
        # Repeat it to match stat rows if both are selected, else just return df_corr
        if include_stat_features:
            df_corr = df_corr.reindex(df_stats.index, method='ffill')
        output_parts.append(df_corr)

    if not output_parts:
        raise ValueError("At least one of `include_stat_features` or `include_corr_matrix` must be True.")

    return pd.concat(output_parts, axis=1)

def laplacian(A):
    D = np.diag(np.sum(A, axis=1))
    return D-A

def inv_laplace(S):
    """
    Convert a normalized Laplacian matrix back to an adjacency matrix.
    """
    # Verify input is square
    assert S.shape[0] == S.shape[1], "Laplacian matrix must be square"
    n = S.shape[0]

    # Verify diagonal is all ones
    assert np.allclose(np.diag(S), np.ones(n)), "Diagonal must be all ones"

    # Create adjacency matrix
    A = np.zeros_like(S)

    # Diagonal elements of adjacency matrix are zero
    np.fill_diagonal(A, 0)

    # Off-diagonal elements: A_ij = -S_ij (since L_ij = -A_ij for i≠j in normalized Laplacian)
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = -S[i, j]
            A[j, i] = A[i, j]  # Symmetry

    return A

def comp_eigen(cov_matrix, n_components=None):
    """Compute eigenvectors and eigenvalues of a sample covariance matrix and sorts them by eigenvalues."""

    # Compute eigenvalues and eigenvectors
    eigenvalues, V_hat = eigh(cov_matrix, lower=True)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V_hat = V_hat[:, idx]

    # Select number of components if specified
    if n_components is not None:
        V_hat = V_hat[:, :n_components]
        eigenvalues = eigenvalues[:n_components]

    # Compute Frobenius norm (total energy)
    E_tot = np.sum(eigenvalues**2)

    return V_hat, eigenvalues, E_tot

def eig_centrality(G, max_attempts=3):
    """
    Compute eigenvector centrality with robust convergence handling.
    """
    # First attempt with default parameters
    try:
        return nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        pass

    # Retry with relaxed parameters
    for attempt in range(1, max_attempts + 1):
        try:
            return nx.eigenvector_centrality(
                G,
                max_iter=5000 * (attempt + 1),
                tol=1e-4 * (attempt + 1)
            )
        except nx.PowerIterationFailedConvergence:
            continue

    # Final fallback to degree centrality
    return nx.degree_centrality(G)

def quadratic_nvar(ts_data, lag=2, alpha=0.5):
    """
    ts_data: np.array of shape (T, N) — time x regions
    Returns: residues from quadratic NVAR model
    """
    T, N = ts_data.shape
    X = []
    Y = []

    # Build lagged design matrix with linear and quadratic terms
    for t in range(lag, T):
        x_lag = ts_data[t - lag]
        linear_terms = x_lag
        quadratic_terms = np.outer(x_lag, x_lag).flatten()
        X.append(np.concatenate([linear_terms, quadratic_terms]))
        Y.append(ts_data[t])

    X = np.array(X)
    Y = np.array(Y)

    # Fit N separate Ridge regressions (or use multivariate regression)
    predictions = []
    for i in range(N):
        model = Ridge(alpha=alpha) # or Lasso
        model.fit(X, Y[:, i])
        predictions.append(model.predict(X))

    predictions = np.array(predictions).T
    residuals = Y - predictions
    return residuals

def laplacian_feats(G):
    L = nx.laplacian_matrix(G).toarray()
    eigvals = np.linalg.eigvalsh(L)  # Sorted real eigenvalues

    # Normalize eigenvalues to compute spectral entropy
    eigvals = np.maximum(eigvals, 1e-12)  # avoid log(0)
    eigvals_norm = eigvals / eigvals.sum()
    spec_entropy = entropy(eigvals_norm)

    # Return useful scalar stats (propagate spectral info, not full eigvecs)
    return {
        "Spectral Entropy": spec_entropy,
        "Mean Laplacian Eigenvalue": np.mean(eigvals),
        "Max Laplacian Eigenvalue": np.max(eigvals),
        "Frobenius Norm (Laplacian Spectrum)": np.linalg.norm(eigvals),
        "Algebraic Connectivity (λ₂)": eigvals[1] if len(eigvals) > 1 else 0
    }

def graph_diameter(G):
    """
    Compute graph diameter with robust handling of disconnected graphs.
    """
    if len(G) == 0:
        return np.nan

    if nx.is_connected(G):
        return nx.diameter(G)
    else:
        try:
            # Get diameter of largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            return nx.diameter(G.subgraph(largest_cc))
        except:
            return np.nan

def detect_communities(G, method='louvain', **kwargs):
    """Detect communities using various algorithms with proper kwargs handling."""
    if method == 'louvain':
        return nx_comm.louvain_communities(G, resolution=kwargs.get('resolution', 1.0))
    elif method == 'greedy_modularity':
        return list(nx_comm.greedy_modularity_communities(G))
    elif method == 'label_propagation':
        return list(nx_comm.label_propagation_communities(G))
    elif method == 'asyn_lpa':  # Asynchronous Label Propagation
        return list(nx_comm.asyn_lpa_communities(G))
    elif method == 'fluid':  # Fluid Communities
        return list(nx_comm.fluid_communities(G, k=kwargs.get('k', 2)))
    elif method == 'girvan_newman':
        return next(nx_comm.girvan_newman(G))  # Returns 2 communities only
    else:
        raise ValueError(f"Unknown method: {method}. Choose: louvain|greedy_modularity|label_propagation|asyn_lpa|fluid|girvan_newman")

def detect_inf_method(ts_data, inf_method, alpha, thresh, cov_method=None):
    """Detect inference method"""
    cov_dependent_methods = {
        'partial_corr',
        'partial_corr',
        'norm_laplacian',
        'rlogspect'
    }

    cov_ignored_methods = {
        'mutual_info',
        'pearson_corr',
        'pearson_corr_binary'
        'gr_causality'
    }

    # Unique warning key
    warning_key = (inf_method, cov_method)

    if inf_method in cov_dependent_methods:
        if cov_method is None:
            raise ValueError(
                f"Method '{inf_method}' requires `cov_method` (choose: 'direct', 'numpy', 'ledoit', 'glasso', 'tv-glasso', 'window', 'var', 'nvar')."
            )
    elif cov_method is not None and inf_method in cov_ignored_methods:
        if warning_key not in _issued_warnings:
            warnings.warn(
                f"`cov_method='{cov_method}'` is ignored for method '{inf_method}'.",
                UserWarning
            )
            _issued_warnings.add(warning_key)

    # Dispatch to methods
    if inf_method == 'sample_cov':
        return sample_covEst(ts_data, cov_method)
    elif inf_method == 'partial_corr':
        C = sample_covEst(ts_data, method=cov_method)
        return partial_corr(C)[0]
    elif inf_method == 'pearson_corr_binary':
        return pearson_corr(ts_data)[0]
    elif inf_method == 'pearson_corr':
        return pearson_corr(ts_data)[1]
    elif inf_method == 'mutual_info':
        return mutual_info(ts_data)
    elif inf_method == 'gr_causality':
        return gr_causality(ts_data)
    elif inf_method == 'norm_laplacian':
        C = sample_covEst(ts_data, method=cov_method)
        S = normalized_laplacian(C, epsilon=0.45, threshold=0)
        return S
    elif inf_method == 'rlogspect':
        C = sample_covEst(ts_data, method=cov_method)
        return learn_adjacency_rLogSpecT(C, delta_n=15*np.sqrt(np.log(176) / 176), threshold=0)
    elif inf_method == 'LADMM':
        C = sample_covEst(ts_data, method=cov_method)
        return learn_adjacency_LADMM(C, delta_n=15*np.sqrt(np.log(176) / 176), threshold=0)
    else:
        raise ValueError(f"Unknown inference method: {inf_method} (choose: 'sample_cov','partial_corr', 'pearson_corr_binary', 'pearson_corr', 'mutual_info', 'gr_causality', 'norm_laplacian', 'rlogspect').")

def sample_covEst(ts_data, method='direct'):
    """Compute the sample covariance estimate using various methods"""
    ts_data = StandardScaler().fit_transform(ts_data) # Standardize data

    if method == 'direct':
        return empirical_covariance(ts_data)
    elif method == 'ledoit':
        lw = LedoitWolf()
        lw.fit(ts_data)
        return lw.covariance_
    elif method == 'glasso':
        gl = GraphicalLasso(alpha=1e-5, max_iter=10000)
        gl.fit(ts_data)
        return gl.covariance_
    elif method == 'tv-glasso':
        w_size = 25  # ~50 seconds for TR=2s
        overlap = 20  # 80% overlap for smoother estimates
        alpha = 1e-5  # Regularization parameter
        T, n_ics = ts_data.shape
        def process_window(window_data):
            gl = GraphicalLasso(alpha=alpha, max_iter=10000)
            gl.fit(window_data - window_data.mean(axis=0))
            return gl.precision_
        covs = Parallel(n_jobs=-1)(
            delayed(process_window)(ts_data[i:i + w_size])
            for i in range(0, T - w_size + 1, w_size - overlap))
        return covs # or mean(covs,axis=0)
    elif method == 'window':
        w_size = 25
        T, n_ics = ts_data.shape
        n_windows = T - w_size + 1
        covs = []
        for i in range(n_windows):
            window = ts_data[i:i+w_size]
            covs.append(np.cov(window, rowvar=False))
        return np.mean(covs, axis=0)
    elif method == 'var':
        ts_df = pd.DataFrame(ts_data)
        results = VAR(ts_df).fit(2)
        #A_list = [results.coefs[i] for i in range(results.k_ar)]  # A_1, A_2, ..., A_p
        sigma_u = results.sigma_u  # Covariance matrix of residuals
        return sigma_u.values
    elif method == 'nvar':
        residuals = quadratic_nvar(ts_data)
        gl = GraphicalLasso(alpha=1e-5, max_iter=10000)
        gl.fit(residuals)
        return gl.covariance_
    else:
        raise ValueError(f"Unknown sample covariance estimation method: {method} (choose: 'direct', 'numpy', 'ledoit', 'glasso', 'tv-glasso', 'window', 'var', 'nvar').")

def threshold_edges(G, alpha=0.2, min_edges=4):
    """Percentile-based thresholding (top 20% of edges by default)"""
    edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
    if not edges:
        return []

    weights = [w for _, _, w in edges]

    # Calculate cutoff as (1-alpha) percentile
    cutoff = np.percentile(weights, 100 * (1 - alpha))

    # Get edges above cutoff
    filtered = [(u, v, w) for u, v, w in edges if w >= cutoff]

    # Ensure minimum edges (fallback to strongest connections)
    if len(filtered) < min_edges:
        filtered = sorted(edges, key=lambda x: x[2], reverse=True)[:min_edges]

    return filtered

def graphing(A, community_method=None, feats=True, plot=False, deg_trh=0, alpha=0.05, min_edges=1):
    """
    Function converting Adjacency matrix to a Graph

    Parameters:
    - It has an input parameter 'super' to select supernodes
    - It has an input parameter 'feats' to compute features
    - It has an input parameter 'plot' to plot the graph
    - deg_thr = degree threshold, removes any nodes with a degree value less than this threshold
    - alpha = sparsity factor, acts as a percentage from the maximum weight used for thresholding

    Note that A is a binary matrix and a type of shift operator(S)
    Therefore A can be modeled as some image data consisting of a pixel grid
    """

    np.fill_diagonal(A, 0) # remove self-loops
    G = nx.from_numpy_array(A)
    G.remove_nodes_from(list(nx.isolates(G)))  # Remove isolated nodes

    if len(G.edges()) < min_edges:
        raise ValueError(f"Graph has only {len(G.edges())} edges (min {min_edges} required)")

    # Use percentile-based edge thresholding
    filtered_data = threshold_edges(G, alpha=alpha, min_edges=min_edges)
    edges_filt = [(u, v) for u, v, _ in filtered_data]
    weights_filt = [w for _, _, w in filtered_data]

    # Compute opacities (normalized weights)
    if weights_filt:
        try:
            min_w = min(weights_filt)
            max_w = max(weights_filt)

            # Handle case where all weights are equal
            if max_w == min_w:
                opacities = [0.5 for _ in weights_filt]  # Set uniform opacity
            else:
                opacities = [0.2 + 0.8 * (w - min_w) / (max_w - min_w) for w in weights_filt]

            # Ensure all opacities are within [0, 1]
            opacities = [max(0.0, min(1.0, op)) for op in opacities]
        except (ValueError, TypeError):
            # Fallback if there are issues with the weights
            opacities = [0.5 for _ in weights_filt]
    else:
        opacities = []

    # Remove nodes with degree < degree threshold
    low_degree_nodes = [node for node, degree in G.degree() if degree < deg_trh]
    G.remove_nodes_from(low_degree_nodes)

    # Supergraph/community mode
    if community_method:
        communities = detect_communities(G, method=community_method)
        communities = [c for c in communities if len(c) >= 2]  # Filter small communities

        # Create supergraph
        superG = nx.Graph()
        node_to_community = {node: i for i, comm in enumerate(communities) for node in comm}

        # Add edges between communities
        inter_edges = set()
        for u, v in G.edges():
            cu, cv = node_to_community.get(u), node_to_community.get(v)
            if cu is not None and cv is not None and cu != cv:
                inter_edges.add(tuple(sorted((cu, cv))))

        superG.add_edges_from(inter_edges)
        G = superG  # Use supergraph for visualization/features
        edges_filt = list(superG.edges())  # Only use superedges
        opacities = [0.6] * len(edges_filt)  # Uniform opacity for superedges

    if plot:
        pos = nx.spring_layout(G, k=0.5, seed=42, iterations=50)
        plt.figure(figsize=(6, 6))

        # Draw nodes and labels
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=100)
        nx.draw_networkx_labels(G, pos, labels={n: n + 1 for n in G.nodes()}, font_weight='bold')

        # Draw only edges that exist in the current graph
        valid_edges = [e for e in edges_filt if e[0] in G.nodes() and e[1] in G.nodes()]
        if valid_edges:
            nx.draw_networkx_edges(G, pos, edgelist=valid_edges, alpha=opacities)
        plt.show()

    # 7. Compute Graph features
    if feats:
        feature_list = []

        # Compute all features
        features = {
            "Degree Centrality": nx.degree_centrality(G),
            "Closeness Centrality": nx.closeness_centrality(G),
            "Eigenvector Centrality": eig_centrality(G),
            "Clustering Coefficient": nx.clustering(G),
            "Average Clustering": nx.average_clustering(G),
            "Edge Betweenness": nx.edge_betweenness_centrality(G),
            "Diameter": graph_diameter(G)
        }

        # Node-level features
        for node in G.nodes():
            node_features = {
                "Node": node,
                "Degree Centrality": features["Degree Centrality"][node],
                "Closeness Centrality": features["Closeness Centrality"][node],
                "Eigenvector Centrality": features["Eigenvector Centrality"][node],
                "Clustering Coefficient": features["Clustering Coefficient"][node]
            }
            feature_list.append(node_features)

        # Graph-level features
        graph_features = {
            "Average Clustering": features["Average Clustering"],
            "Diameter": features["Diameter"],
            **laplacian_feats(G),  # Spectral features
            # "Global Efficiency": nx.global_efficiency(G),
            "Graph Energy": np.sum(np.abs(np.linalg.eigvalsh(A)))
        }

        # Create DataFrames for nodes and graph features
        node_df = pd.DataFrame(feature_list)
        node_df['ROI'] = node_df['Node'].apply(lambda x: f'ROI_{x + 1}')  # Label nodes as ROIs
        node_df = node_df.drop(columns=['Node'])
        roi_col = node_df.pop('ROI')
        node_df.insert(0, 'ROI', roi_col)
        graph_df = pd.DataFrame([graph_features])

        # --- New: Edge-level features ---
        # Extract edges with weights from the graph
        edges_data = []
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1.0)  # If no weight attribute, default to 1.0
            edges_data.append({
                "Node1": f'ROI_{u + 1}',
                "Node2": f'ROI_{v + 1}',
                "EdgeWeight": weight
            })

        edge_df = pd.DataFrame(edges_data)
        return node_df, graph_df

def adj_heatmap(W):
    plt.figure(figsize=(8, 6))
    sns.heatmap(W,
                cmap='coolwarm',
                square=True)
    plt.xlabel("ROI")
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.show()

def pearson_corr(data, absolute_value=True):
    """
    Compute Pearson correlation graph (fully connected).

    Returns:
    - adjacency matrix (binary, excluding diagonal)
    - correlation matrix
    """
    corr_matrix = np.corrcoef(data.T)

    if absolute_value:
        corr_matrix = np.abs(corr_matrix)

    np.fill_diagonal(corr_matrix, 1.0)

    # Fully connected graph (excluding self-loops)
    N = data.shape[1]
    adj_matrix = np.ones((N, N)) - np.eye(N)

    return adj_matrix, corr_matrix

def partial_corr(cov_matrix):
    """
    Compute partial correlation matrix from covariance.
    """
    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_matrix)

    diag = np.sqrt(np.diag(inv_cov))
    W = -inv_cov / np.outer(diag, diag)
    np.fill_diagonal(W, 0)

    return W, inv_cov

def mutual_info(data, n_jobs=-1):
    """
    Compute mutual information matrix using joblib for parallelism.
    """
    def compute_pairwise_mi(data, i, j):
        """
        Compute mutual information between two columns.
        """
        x = data[:, i]
        y = data[:, j]
        return i, j, mutual_info_regression(x.reshape(-1, 1), y, discrete_features=False)[0]

    N = data.shape[1]
    mi_matrix = np.zeros((N, N))

    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_pairwise_mi)(data, i, j) for i, j in pairs
    )

    for i, j, mi in results:
        mi_matrix[i, j] = mi_matrix[j, i] = mi

    return mi_matrix

def gr_causality(data, max_lag=5, n_jobs=-1, verbose=False):
    # Initialize
    N = data.shape[1]
    gr_matrix = np.zeros((N, N), dtype=np.float32)

    # Pre-allocate test pairs
    test_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]

    def _granger_test(i, j):
        """Inner function for parallel execution"""
        with contextlib.redirect_stdout(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            try:
                test_result = grangercausalitytests(
                    data[:, [j, i]],
                    maxlag=max_lag,
                    verbose=False  # Explicitly set to False
                )
                return max(test_result[lag][0]['ssr_ftest'][0] for lag in range(1, max_lag + 1))
            except Exception:
                return 0.0

    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(_granger_test)(i, j)
        for i, j in tqdm(test_pairs, disable=not verbose))

    # Fill matrix
    idx = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                gr_matrix[i, j] = results[idx]
                idx += 1

    return gr_matrix

def load_files(folder_path=None, var_filt=True, ica=False, sex='all', site=None, max_files=None, shuffle=False, n_components=20):
    """
    Load .1D fMRI files with options to filter by gender, limit number of files, and shuffle data.

    Parameters:
    - folder_path: base folder path. Defaults to '.../abide'.
    - var_filt: apply zero variance filter.
    - ica: apply ICA dimensionality reduction.
    - gender: 'male', 'female', or 'all'
    - max_files: maximum number of files to load
    - shuffle: whether to shuffle data and subject IDs

    Returns:
    - all_data: List of 2D numpy arrays (one per file)
    - subject_ids: List of subject IDs
    - file_list: List of file paths
    - file_info: Dictionary with metadata
    """
    if folder_path is None:
        folder_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "abide")

    # Find sex-specific subfolders dynamically (e.g., male-cpac-..., female-cpac-...)
    subdirs = [d for d in glob.glob(os.path.join(folder_path, '*')) if os.path.isdir(d)]

    sex_folders = []
    if sex.lower() in ['male', 'female']:
        sex_folders = [d for d in subdirs if os.path.basename(d).lower().startswith(sex.lower())]
    elif sex.lower() == 'all':
        sex_folders = [d for d in subdirs if os.path.basename(d).lower().startswith(('male', 'female'))]
    else:
        raise ValueError("sex must be 'male', 'female', or 'all'")

    file_list = []
    for sfolder in sex_folders:
        file_list.extend(glob.glob(os.path.join(sfolder, '*.1D')))

    if site is not None:
        file_list = [f for f in file_list if os.path.basename(f).split('_005')[0] == site]

    if shuffle:
        random.shuffle(file_list)
    else:
        file_list = sorted(file_list)

    if max_files is not None:
        file_list = file_list[:max_files]

    if not file_list:
        raise ValueError(f"No .1D files found for sex='{sex}' in {folder_path}.")

    all_data = []
    subject_ids = []
    site_ids = []
    loaded_files = []
    file_info = {
        'total_files': 0,
        'timepoints_per_file': [],
        'series_per_file': []
    }

    for file_path in file_list:
        try:
            data = np.loadtxt(file_path)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            all_data.append(data)
            filename = os.path.basename(file_path)
            subject_id = '005' + filename.split('_005')[1].split('_')[0]
            site_id = filename.split('_005')[0]
            subject_ids.append(subject_id)
            site_ids.append(site_id)
            loaded_files.append(file_path)

            file_info['timepoints_per_file'].append(data.shape[0])
            file_info['series_per_file'].append(data.shape[1])
            file_info['total_files'] += 1

            #print(f"Loaded {filename}: {data.shape[0]} timepoints × {data.shape[1]} series")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    if var_filt:
        all_data, subject_ids, _ = zeroVar_filter(all_data, subject_ids)

    if ica:
        all_data = ica_smith(all_data, n_components=n_components)


    return all_data, subject_ids, loaded_files, file_info

def ica_dimReduc(fmri_data, subject_ids=None, n_components=15, max_iter=10000, tol=1e-5,
                prinCA=True, ica_attempts=3, remove_failures=True):
    """
    Reduce dimensionality of fMRI time series data using ICA with:
    - Multiple convergence attempts (ica_attempts, 3 standard)
    - Fallback to PCA
    - Numerical stability checks

    Parameters:
    - fmri_data: List of subject time series data (each subject: timepoints × 116 AAL regions)
    - n_components: Number of ICA components to extract (default: 30)

    Returns:
    - ica_components: List of ICA components per subject (timepoints × n_components)
    - mixing_matrices: List of mixing matrices per subject (116 regions × n_components)
    - filtered_ids: Subject IDs that passed ICA (only if subject_ids provided)
    - failed_indices: Indices of removed subjects
    """
    scaler = StandardScaler()
    ica_components = []
    mixing_matrices = []
    successful_indices = []

    for subj_idx, subject in enumerate(fmri_data):
        print(f"\nSubject {subj_idx + 1}/{len(fmri_data)}")
        X = scaler.fit_transform(subject)

        # 1. Data Quality Check
        valid_mask = np.var(X, axis=0) > 1e-8
        if np.sum(valid_mask) < n_components:
            print(f"Only {np.sum(valid_mask)} valid ROIs - skipping")
            continue

        X = X[:, valid_mask]

        # 2. PCA Whitening with Rank Check
        if prinCA:
            pca = PCA(n_components=min(n_components, X.shape[1] - 1), whiten=True)
            try:
                X_white = pca.fit_transform(X)
                explained_var = np.sum(pca.explained_variance_ratio_)
                print(f"PCA explained variance: {explained_var:.1%}")
                if explained_var < 0.5:
                    raise ValueError("PCA explains <50% variance")
            except Exception as e:
                print(f"PCA failed: {str(e)}")
                if remove_failures: continue
                X_white = X[:, :n_components]  # Fallback

        # 3. ICA with Enhanced Attempts
        success = False
        for attempt in range(ica_attempts):
            try:
                ica = FastICA(
                    n_components=n_components,
                    max_iter=max_iter,
                    tol=tol,
                    random_state=attempt * 100 + 42,  # Diverse seeds
                    algorithm='parallel',
                )

                components = ica.fit_transform(X_white if prinCA else X)

                # Component Quality Check
                if np.max(np.abs(components)) < 1e-6:
                    raise ValueError("Near-zero components")

                # Store results
                mixing = ica.mixing_ if hasattr(ica, 'mixing_') else pinv(ica.components_)
                if prinCA:
                    mixing = pca.components_.T @ mixing

                ica_components.append(components)
                mixing_matrices.append(mixing)
                successful_indices.append(subj_idx)
                print(f"ICA succeeded in {ica.n_iter_} iterations")
                success = True
                break

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")

        if not success and not remove_failures:
            print("Using PCA fallback")
            ica_components.append(X_white if prinCA else X[:, :n_components])
            mixing_matrices.append(pca.components_.T if prinCA else np.eye(X.shape[1], n_components))
            successful_indices.append(subj_idx)

    # Handle subject IDs
    filtered_ids = [subject_ids[i] for i in successful_indices] if subject_ids else None
    failed_indices = [i for i in range(len(fmri_data)) if i not in successful_indices]

    if failed_indices:
        print(f"\nFailed on {len(failed_indices)} subjects: {failed_indices}")

    return ica_components, mixing_matrices, filtered_ids, failed_indices

def ica_smith(
        aal_time_series_list: List[np.ndarray],
        n_components: int = 20,
        standardize: bool = False,
        return_transformation_matrix: bool = False
) -> Union[List[np.ndarray], tuple]:
    """
    Converts AAL time series (116 ROIs) to Smith 2009 ICA component time series (10/20/70 RSNs).

    Parameters:
    -----------
    n_components, to select number of ICA components
    """
    # --- Input Validation ---
    valid_components = {10, 20, 70}
    if n_components not in valid_components:
        raise ValueError(f"n_components must be one of {valid_components}, got {n_components}")

    for i, ts in enumerate(aal_time_series_list):
        if ts.shape[1] != 116:
            raise ValueError(f"Subject {i} has {ts.shape[1]} ROIs (expected 116). Transpose?")

    # --- Load Smith 2009 ICA Maps and AAL Atlas ---
    smith = datasets.fetch_atlas_smith_2009()

    # Select the appropriate ICA map based on n_components
    if n_components == 10:
        smith_maps = image.load_img(smith.rsn10)
    elif n_components == 20:
        smith_maps = image.load_img(smith.rsn20)
    else:  # 70 components
        smith_maps = image.load_img(smith.rsn70)

    aal = datasets.fetch_atlas_aal()
    aal_img = image.load_img(aal.maps)  # AAL atlas

    # --- Compute ICA-to-AAL Transformation Matrix ---
    n_regions = 116
    ica_to_aal = np.zeros((n_regions, n_components))

    masker = NiftiLabelsMasker(labels_img=aal_img, standardize=False)
    for i in range(n_components):
        component_img = image.index_img(smith_maps, i)
        signals = masker.fit_transform(component_img)
        ica_to_aal[:, i] = signals[0]  # Spatial map for ICA component i

    # Pseudo-inverse for projection (AAL → ICA space)
    ica_to_aal_inv = np.linalg.pinv(ica_to_aal)  # Shape: [n_components × 116]

    # --- Project Each Subject's Data ---
    smith_ics_list = []
    for ts in aal_time_series_list:
        if standardize:
            ts = (ts - ts.mean(axis=0)) / (ts.std(axis=0) + 1e-10)  # Z-score with epsilon to avoid division by zero
        smith_ics = ts @ ica_to_aal_inv.T  # [time × 116] @ [116 × n_components] → [time × n_components]
        smith_ics_list.append(smith_ics)

    # --- Return Results ---
    if return_transformation_matrix:
        return smith_ics_list, ica_to_aal_inv
    else:
        return smith_ics_list

def group_ica(data_list, n_components=30):
    """
    Perform group ICA (GIG-ICA approximation) on a list of AAL time series arrays.

    Parameters:
    - data_list: list of np.ndarray, each with shape (T, 116)
    - n_components: number of ICA components

    Returns:
    - group_components: group-level ICs (n_components x features)
    - individual_timecourses: list of individual subject component time series (T x n_components)
    """
    # Step 1: Z-score each subject's time series across time
    standardized = [StandardScaler().fit_transform(ts) for ts in data_list]

    # Step 2: Concatenate all subjects' time series along time axis
    concat_data = np.vstack(standardized)  # shape (sum(T), 116)

    # Step 3: Group ICA using FastICA
    group_ica = FastICA(n_components=n_components, max_iter=1000, random_state=42)
    group_sources = group_ica.fit_transform(concat_data)  # shape (sum(T), n_components)

    # Step 4: Unmixing matrix for individual back-reconstruction
    unmixing_matrix = group_ica.components_  # shape (n_components, 116)

    # Step 5: Back-reconstruct individual timecourses
    lengths = [ts.shape[0] for ts in data_list]
    cumulative = np.cumsum([0] + lengths)

    individual_timecourses = []
    for i in range(len(data_list)):
        subject_data = standardized[i]  # shape (T, 116)
        subject_sources = subject_data @ unmixing_matrix.T  # shape (T, n_components)
        individual_timecourses.append(subject_sources)

    return group_ica.components_, individual_timecourses

def multiset_feats(data_list, subject_ids, inf_method="mutual_info", cov_method=None,
                   thresh=0.2, n_jobs=-1, feats="graph"):
    """
    Parallelized version of subject-wise feature extraction.

    Parameters:
    -----------
    - inf_method, graph inference method used: {'partial_corr', 'pearson_corr_binary', 'pearson_corr', 'mutual_info', 'gr_causality', 'norm_laplacian', 'rlogspect'}
    - cov_method, sample covariance estimation method used: {'direct', 'numpy', 'ledoit', 'glasso', 'window', 'var'}
    - feats, select which features to compute: {'both', 'stat', 'graph'}
    """

    valid_options = ["both", "stat", "graph"]
    if feats not in valid_options:
        raise ValueError(f"compute_features must be one of {valid_options}")

    def process_subject(data, sid, inf_method, cov_method):
        """
        Processes a single subject's data with feature selection.
        """
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            # Initialize empty DataFrames
            df_stat = pd.DataFrame()
            df_graph = pd.DataFrame()
            df_global = pd.DataFrame()

            # Compute statistical features if requested
            if feats in ["both", "stat"]:
                df_stat = stat_feats(data)

            # Compute graph features if requested
            if feats in ["both", "graph"]:
                try:
                    adj_matrix = detect_inf_method(data, inf_method=inf_method, cov_method=cov_method)
                    if np.all(adj_matrix == 0):
                        raise ValueError("Zero adjacency matrix")

                    df_graph, df_global = graphing(adj_matrix, alpha=thresh)

                    # If computing both, merge stat and graph features
                    if feats == "both":
                        df_roi = pd.merge(df_stat, df_graph, on='ROI', how='left')
                    else:
                        df_roi = df_graph

                except Exception as graph_err:
                    print(f"Graph failed for {sid}: {str(graph_err)}")
                    if feats == "both":
                        df_roi = df_stat  # Fall back to stat features only
                    else:
                        df_roi = pd.DataFrame()  # Empty if only graph requested

            # Handle case where only stat features are requested
            elif feats == "stat":
                df_roi = df_stat

            # Add subject ID if we have any features
            if not df_roi.empty:
                df_roi['subject_id'] = sid

            return df_roi, df_global

        except Exception as e:
            print(f"Subject {sid} failed completely: {str(e)}")
            return None, None

    # Parallel processing (unchanged)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(data, sid, inf_method, cov_method)
        for data, sid in tqdm(zip(data_list, subject_ids),
                              total=len(subject_ids),
                              desc="Processing subjects")
    )

    # Filter successful results
    df_app_list, df_global_list = zip(*[(df_roi, df_global) for (df_roi, df_global) in results if df_roi is not None])

    # Handle case where no graph features were computed
    if feats != "graph":
        df_global_list = [df for df in df_global_list if not df.empty]

    # Fast concatenation without sorting columns
    df_app = pd.concat(df_app_list, ignore_index=True, sort=False)
    df_global = pd.concat(df_global_list, ignore_index=True, sort=False) if df_global_list else pd.DataFrame()

    # Drop columns that are entirely NaN
    df_app.dropna(axis=1, how='all', inplace=True)

    # Skip pivoting if only global features were requested
    if not df_app.empty:
        # Pivot to wide format
        df_wide = df_app.pivot_table(
            index='subject_id',
            columns='ROI',
            values=df_app.columns.difference(['ROI', 'subject_id']),
            aggfunc='first'
        )
        # Flatten MultiIndex columns
        df_wide.columns = [f"{stat}_{roi}" for stat, roi in df_wide.columns]
        df_wide.reset_index(inplace=True)

        # Sort feature columns
        def extract_roi_num(col):
            match = re.search(r'_(\d+)$', col)
            return int(match.group(1)) if match else -1

        id_cols = ['subject_id']
        feature_cols = [col for col in df_wide.columns if col not in id_cols]
        features_grouped = sorted(feature_cols, key=lambda x: (extract_roi_num(x), x.split('_')[0]))
        df_wide = df_wide[id_cols + features_grouped]
    else:
        df_wide = pd.DataFrame({'subject_id': subject_ids})

    # Join with global features if they exist
    if not df_global.empty:
        df_final = pd.concat([df_wide, df_global.reset_index(drop=True)], axis=1)
    else:
        df_final = df_wide

    # Append phenotypic data
    def multiset_pheno(df_wide):
        # Fix the path construction
        base_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.join(base_path, "..", "..")
        pheno_path = os.path.join(parent_path, "abide", "Phenotypic_V1_0b_preprocessed1.csv")

        df_labels = pd.read_csv(pheno_path)

        df_labels['SUB_ID'] = df_labels['SUB_ID'].astype(str).str.zfill(7)
        df_wide['subject_id'] = df_wide['subject_id'].astype(str).str.zfill(7)

        df_pheno = df_labels[['SUB_ID', 'SITE_ID', 'DX_GROUP', 'SEX']]
        df_merged = df_wide.merge(df_pheno, left_on='subject_id', right_on='SUB_ID', how='left')
        df_merged.drop(columns='SUB_ID', inplace=True)

        phenotype_cols = ['DX_GROUP', 'SEX', 'SITE_ID']
        cols = phenotype_cols + [col for col in df_merged.columns if col not in phenotype_cols]
        df_merged = df_merged[cols]

        return df_merged

    return multiset_pheno(df_final)

def adjacency_df(data_list, subject_ids, inf_method, cov_method, alpha, thresh):
    rows = []
    index = []

    for i, (data, sid) in enumerate(zip(data_list, subject_ids)):
        try:
            #print(f"[{i+1}/{len(data_list)}] Processing subject {sid}...", flush=True)
            adj_matrix = detect_inf_method(data, inf_method=inf_method, cov_method=cov_method, alpha=alpha, thresh=thresh)
            if adj_matrix is None or np.all(adj_matrix == 0):
                print(f" - Empty or zero matrix for subject {sid}. Skipping.")
                continue
            flat_adj = adj_matrix.flatten()
            rows.append(flat_adj)
            index.append(sid)
        except Exception as e:
            print(f" - Failed for subject {sid}: {str(e)}")
            continue

    if not rows:
        raise RuntimeError("No valid adjacency matrices were computed.")

    n = data_list[0].shape[1]
    col_names = [f"A_{i}_{j}" for i in range(n) for j in range(n)]
    df_wide =  pd.DataFrame(rows, index=index, columns=col_names).reset_index().rename(columns={'index': 'subject_id'})
    
    def multiset_pheno(df_wide):
        # Fix the path construction
        base_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.join(base_path, "..", "..")
        pheno_path = os.path.join(parent_path, "abide", "Phenotypic_V1_0b_preprocessed1.csv")

        df_labels = pd.read_csv(pheno_path)

        df_labels['SUB_ID'] = df_labels['SUB_ID'].astype(str).str.zfill(7)
        df_wide['subject_id'] = df_wide['subject_id'].astype(str).str.zfill(7)

        # Select desired phenotypic columns
        df_pheno = df_labels[['SUB_ID', 'SITE_ID', 'DX_GROUP', 'SEX']]

        # Merge and drop SUB_ID
        df_merged = df_wide.merge(df_pheno, left_on='subject_id', right_on='SUB_ID', how='left')
        df_merged.drop(columns='SUB_ID', inplace=True)

        # Reorder phenotypic columns
        phenotype_cols = ['DX_GROUP', 'SEX', 'SITE_ID']
        cols = phenotype_cols + [col for col in df_merged.columns if col not in phenotype_cols]
        df_merged = df_merged[cols]

        return df_merged

    return multiset_pheno(df_wide)

#-------{Main for testing}-------#
# fmri_data, subject_ids, file_paths, metadata = load_files() # represents format of load_files()
# output_df = multiset_feats(fmri_data,subject_ids)           # represents multiset_feats() usage, add another index '[]' to load_files to select data amount

#fmri_data, subject_ids, _, _ = load_files(sex='all', site='NYU', max_files=10, shuffle=True, var_filt=False, ica=True)

#print("fmri_data_shape: " + str(len(fmri_data)))

#df_out = multiset_feats(fmri_data, subject_ids, inf_method='rlogspect', cov_method='glasso',feats='graph')

#print("df_out:\n", df_out)
#print("Feature list: ",list(df_out.columns))
