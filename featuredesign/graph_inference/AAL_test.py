import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os,glob,re,random
import pandas as pd
import seaborn as sns
import networkx.algorithms.community as nx_comm

from scipy.stats import pearsonr,skew, kurtosis
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV, LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations
from scipy.signal import find_peaks
from sklearn.decomposition import FastICA,PCA
from scipy.linalg import pinv, eigh
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.tsatools import detrend
from nilearn import datasets, image
from nilearn.input_data import NiftiLabelsMasker
from typing import List, Union

from featuredesign.graph_inference.GSP_methods import normalized_laplacian, normalized_laplacian_reweighted, adjacency_reweighted, learn_adjacency_rLogSpecT

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

def detect_inf_method(ts_data, method):
    """Detect inference method"""
    if method == 'partial_corr_LF':
        return partial_corr(ts_data, threshold=None)[0]
    elif method == 'partial_corr_glasso':
        return partial_corr(ts_data, method="glasso")[0]
    elif method == 'pearson_corr_binary':
        return pearson_corr(ts_data)[0]
    elif method == 'pearson_corr':
        return pearson_corr(ts_data)[1]
    elif method == 'mutual_info':
        return mutual_info(ts_data)
    elif method == 'norm_laplacian':
        C = sample_covEst(ts_data)
        V_hat,_,_ = comp_eigen(C)
        S = normalized_laplacian(V_hat)
        if S is None:
            raise ValueError("Laplacian matrix is None. Check prior steps for subject.")
        return inv_laplace(S)
    elif method == 'rlogspect':
        C = sample_covEst(ts_data)
        V_hat,_,E_tot = comp_eigen(C)
        return learn_adjacency_rLogSpecT(V_hat,delta_n=0.2*np.sqrt(E_tot)) # threshold dn to remove noisy eigenvalues
    else:
        raise ValueError(f"Unknown method: {method}. Choose: partial_corr_LF|partial_corr_glasso|pearson_corr_binary|pearson_corr|mutual_info|norm_laplacian|rlogspect")

def sample_covEst(ts_data, method='glasso'):
    """Compute the sample covariance estimate using various methods"""
    if method == 'direct':
        T, n_ics = ts_data.shape
        X_centered = ts_data - np.mean(ts_data, axis=0)
        cov = (X_centered.T @ X_centered) / (T - 1)
        return cov
    elif method == 'ledoit':
        lw = LedoitWolf()
        lw.fit(ts_data)
        return lw.covariance_
    elif method == 'glasso':
        gl = GraphicalLasso(alpha=1e-5, max_iter=10000)
        gl.fit(ts_data)
        return gl.covariance_
    elif method == 'window':
        window_size = 25
        T, n_ics = ts_data.shape
        n_windows = T - window_size + 1
        covs = []
        for i in range(n_windows):
            window = ts_data[i:i+window_size]
            covs.append(np.cov(window, rowvar=False))
        return np.mean(covs, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}. Choose: direct|ledoit|glasso|window")

def stat_feats(x):
    """
    Compute classic statistical features based on some time series data and stores it to a panda dataframe.

    Features:
    - Mean, Standard Deviation, Skewness, Kurtosis, Slope, Correlation, Covariance, Signal-to-noise ratio etc.
    """
    # Transpose back so each row is one ROI's time series
    if x.shape[0] > x.shape[1]:
        x = x.T

    # Extract features for each time series
    feature_list = []
    for ts in x:
        features = {
            'mean': np.mean(ts, axis=0),
            'std': np.std(ts, axis=0),
            'SNR': np.mean(ts, axis=0) / (np.std(ts, axis=0) + 1e-10),
            'Skewness': skew(ts, axis=0),
            'Kurtosis': kurtosis(ts, axis=0)
        }
        feature_list.append(features)

    # Compute peak statistics
    df_pk = pk_stats(x)
    df = pd.DataFrame(feature_list)
    df = pd.concat([df, df_pk], axis=1)
    # Generate ROI labels
    aal_labels = [f"ROI_{i + 1}" for i in range(len(df))]
    df.insert(0, 'ROI', aal_labels)

    return df

def graphing(A, community_method=None, feats=True, plot=False, deg_trh=0, alpha=0e-0, min_edges=1):
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

    # Extract edge weights and normalize for linewidth
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # Filter weights to promote sparsity (=weight/edge removal)
    filtered_data = [(u, v, G[u][v]['weight']) for u, v in edges if G[u][v]['weight'] >= alpha * max(weights)]
    edges_filt = [(u, v) for u, v, w in filtered_data]  # Just the edges
    weights_filt = [w for u, v, w in filtered_data]  # Just the weights

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
        G.remove_nodes_from(list(nx.isolates(G)))  # Remove isolated nodes
        feature_list = []

        # Compute all features
        features = {
            "Degree Centrality": nx.degree_centrality(G),
            "Closeness Centrality": nx.closeness_centrality(G),
            "Eigenvector Centrality": eig_centrality(G),
            "Clustering Coefficient": nx.clustering(G),
            "Shortest Paths": dict(nx.all_pairs_shortest_path_length(G)),
            "Average Clustering": nx.average_clustering(G),
            "Edge Betweenness": nx.edge_betweenness_centrality(G),
            "Diameter": graph_diameter(G),
            "Laplacian Eigenvectors": np.linalg.eig(laplacian(A))[0].real

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
            "Laplacian Eigenvectors": features["Laplacian Eigenvectors"] # is not really a graph feature, more so a temporal/spectral feature (its says something about the frequency components)
        }

        # Create DataFrames
        node_df = pd.DataFrame(feature_list)
        node_df['ROI'] = node_df['Node'].apply(lambda x: f'ROI_{x + 1}')    # Set nodes equal to ROIs in dataframe and remove nodes index
        node_df = node_df.drop(columns=['Node'])
        roi_col = node_df.pop('ROI')
        node_df.insert(0, 'ROI', roi_col)
        graph_df = pd.DataFrame([graph_features])

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

def pearson_corr(time_series_data, threshold=0.5, absolute_value=True):
    """
    Build a graph based on Pearson correlation between time series.

    Parameters:
    - time_series_data: 2D numpy array or pandas DataFrame (n_timepoints × n_series)
    - threshold: correlation threshold for edge creation
    - absolute_value: if True, uses absolute value of correlation

    Returns:
    - adjacency matrix of the graph
    - correlation matrix
    """
    n_series = time_series_data.shape[1]
    corr_matrix = np.zeros((n_series, n_series))
    adj_matrix = np.zeros((n_series, n_series))

    # Calculate pairwise correlations
    for i in range(n_series):
        for j in range(i, n_series):  # Include diagonal for corr_matrix
            x = time_series_data[:, i]
            y = time_series_data[:, j]

            # Handle constant series cases
            if i == j:
                corr = 1.0  # Diagonal
            elif np.all(x == x[0]) or np.all(y == y[0]):
                corr = 0.0  # Constant series
            else:
                corr = pearsonr(x, y)[0]

            if absolute_value:
                corr = abs(corr)

            # Threshold for adjacency (skip diagonal)
            if i != j and corr > threshold:
                adj_matrix[i, j] = adj_matrix[j, i] = 1
                corr_matrix[i, j] = corr

    return adj_matrix, corr_matrix

def partial_corr(ts_data, method="glasso", threshold=None):
    # Standardize the data
    ts_data = StandardScaler().fit_transform(ts_data)

    # Compute inverse covariance
    if method == "glasso":
        inv_cov = GraphicalLassoCV(max_iter=5000).fit(ts_data).precision_
    else:
        inv_cov = LedoitWolf().fit(ts_data).precision_

    # Compute partial correlations
    diag = np.sqrt(np.diag(inv_cov))
    W = -inv_cov / np.outer(diag, diag)
    np.fill_diagonal(W, 0)

    # Apply threshold if specified
    if threshold is not None:
        if 0 < threshold < 1:  # Percentile threshold (0-100)
            abs_vals = np.abs(W[np.triu_indices_from(W, k=1)])
            threshold = np.percentile(abs_vals, threshold * 100)
        W[np.abs(W) < threshold] = 0

    return W, inv_cov

def mutual_info(data):
    N = data.shape[1]
    mi_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            # sklearn's MI expects 2D input for X and 1D for y
            mi = mutual_info_regression(data[:, i].reshape(-1, 1), data[:, j])[0]
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    return mi_matrix

def load_files(folder_path=None, var_filt=True, ica=False, sex='all', max_files=None, shuffle=False):
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
            subject_ids.append(subject_id)
            loaded_files.append(file_path)

            file_info['timepoints_per_file'].append(data.shape[0])
            file_info['series_per_file'].append(data.shape[1])
            file_info['total_files'] += 1

            print(f"Loaded {filename}: {data.shape[0]} timepoints × {data.shape[1]} series")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    if var_filt:
        all_data, subject_ids, _ = zeroVar_filter(all_data, subject_ids)

    if ica:
        all_data = ica_smith(all_data)


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
        standardize: bool = False,
        return_transformation_matrix: bool = False
) -> Union[List[np.ndarray], tuple]:
    """
    Converts AAL time series (116 ROIs) to Smith 2009 ICA component time series (10 RSNs).

    Parameters:
    -----------
    aal_time_series_list : List[np.ndarray]
        List of 2D arrays, each with shape [n_timepoints × 116] (AAL time series per subject).
    standardize : bool, optional (default=False)
        If True, standardizes each subject's time series (z-score) before projection.
    return_transformation_matrix : bool, optional (default=False)
        If True, returns the pseudo-inverse transformation matrix (ica_to_aal_inv).

    Returns:
    --------
    smith_ics_list : List[np.ndarray]
        List of 2D arrays, each with shape [n_timepoints × 10] (Smith IC time series per subject).
    ica_to_aal_inv : np.ndarray (optional)
        Pseudo-inverse transformation matrix [10 × 116], returned only if return_transformation_matrix=True.
    """
    # --- Input Validation ---
    for i, ts in enumerate(aal_time_series_list):
        if ts.shape[1] != 116:
            raise ValueError(f"Subject {i} has {ts.shape[1]} ROIs (expected 116). Transpose?")

    # --- Load Smith 2009 ICA Maps and AAL Atlas ---
    smith = datasets.fetch_atlas_smith_2009()
    smith_maps = image.load_img(smith.rsn10)  # 10 ICA components

    aal = datasets.fetch_atlas_aal()
    aal_img = image.load_img(aal.maps)  # AAL atlas

    # --- Compute ICA-to-AAL Transformation Matrix ---
    n_components = smith_maps.shape[-1]  # 10
    n_regions = 116
    ica_to_aal = np.zeros((n_regions, n_components))

    masker = NiftiLabelsMasker(labels_img=aal_img, standardize=False)
    for i in range(n_components):
        component_img = image.index_img(smith_maps, i)
        signals = masker.fit_transform(component_img)
        ica_to_aal[:, i] = signals[0]  # Spatial map for ICA component i

    # Pseudo-inverse for projection (AAL → ICA space)
    ica_to_aal_inv = np.linalg.pinv(ica_to_aal)  # Shape: [10 × 116]

    # --- Project Each Subject's Data ---
    smith_ics_list = []
    for ts in aal_time_series_list:
        if standardize:
            ts = (ts - ts.mean(axis=0)) / ts.std(axis=0)  # Z-score
        smith_ics = ts @ ica_to_aal_inv.T  # [time × 116] @ [116 × 10] → [time × 10]
        smith_ics_list.append(smith_ics)

    # --- Return Results ---
    if return_transformation_matrix:
        return smith_ics_list, ica_to_aal_inv
    else:
        return smith_ics_list

def multiset_feats(data_list, subject_ids, method="mutual_info"):
    """
    Loops over all data recursively to compute specific features and stores them in a dataframe indexed per individual (subject ID).

    - Input: fmri_dataset of 1D timeseries, corresponding subject_ids
    - Output: DataFrame indexed per individual with all relative features
    - Parameters: 'method' to choose graph inference method used

    """
    df_app = pd.DataFrame()
    df_global_list = []
    expanded_ids = []

    for data, sid in zip(data_list, subject_ids):
        try:
            df_stat = stat_feats(data)

            try:
                adj_matrix = detect_inf_method(data, method=method)
                if np.all(adj_matrix == 0):
                    raise ValueError("Zero adjacency matrix")

                df_graph, df_global = graphing(adj_matrix)
                df_roi = pd.merge(df_stat, df_graph, on='ROI', how='left')

            except Exception as graph_err:
                print(f"Graph failed for {sid}: {str(graph_err)}")
                df_roi = df_stat  # Keep statistical features only
                df_global = pd.DataFrame()  # Empty global features

            # Accumulate results
            df_global_list.append(df_global)
            expanded_ids.extend([sid] * len(df_roi)) # Pad subject IDs with copies for all ROIs
            df_app = pd.concat([df_app, df_roi], ignore_index=True)

        except Exception as e:
            print(f"Subject {sid} failed completely: {str(e)}")
            continue

    df_global = pd.concat(df_global_list, ignore_index=True)

    # Assign to dataframe
    df_app['subject_id'] = expanded_ids
    df_app = df_app.dropna(axis=1)  # Remove NaN columns

    # Automatically pivots all feature columns
    df_wide = df_app.pivot(
        index=['subject_id'],
        columns='ROI',
        values=df_app.columns.difference(['ROI', 'subject_id'])
    )

    # Flatten column names
    df_wide.columns = [f"{stat}_{roi}" for stat, roi in df_wide.columns]
    df_wide = df_wide.reset_index()

    # Function to extract ROI number for sorting
    def extract_roi_num(col):
        match = re.search(r'_(\d+)$', col)
        return int(match.group(1)) if match else -1

    # Sort columns: keep subject_id and institute first, then group by ROI
    cols = df_wide.columns.tolist()
    id_cols = ['subject_id']
    feature_cols = [col for col in cols if col not in id_cols]

    # Group by ROI number
    features_grouped = sorted(feature_cols, key=lambda x: (extract_roi_num(x), x.split('_')[0]))
    df_wide = pd.concat([df_wide[id_cols + features_grouped], df_global], axis=1)

    def multiset_pheno(df_wide):
        """
        Function that loads phenotypic data and merges it into the input dataframe.

        - DX_GROUP: 1=ASD,2=ALL
        - SEX: 1=Male,2=Female

        """
        df_labels = pd.read_csv(
            os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "abide",
                         "Phenotypic_V1_0b_preprocessed1.csv"))

        # Convert SUB_ID to match subject_id format
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
"""
fmri_data, subject_ids, _, _ = load_files(sex='all', max_files=800, shuffle=True, var_filt=True, ica=True)

print(f"Final data: {len(fmri_data)} subjects")
print(f"Final IDs: {len(subject_ids)}")

df_out = multiset_feats(fmri_data, subject_ids, method='rlogspect')

print("df_out:\n", df_out)
"""