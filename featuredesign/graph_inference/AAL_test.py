import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import re
import seaborn as sns
import networkx.algorithms.community as nx_comm

from scipy.stats import pearsonr,skew, kurtosis
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV, LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations
from scipy.signal import find_peaks

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

def detect_inf_method(data, method):
    """Detect inference method"""
    if method == 'partial_corr_LF':
        return partial_corr(data)[0]
    elif method == 'partial_corr_glasso':
        return partial_corr(data, method="glasso")[0]
    elif method == 'pearson_corr_binary':
        return pearson_corr(data)[0]
    elif method == 'pearson_corr':
        return pearson_corr(data)[1]
    elif method == 'mutual_info':
        return mutual_info(data)
    else:
        raise ValueError(f"Unknown method: {method}. Choose: partial_corr_LF|partial_corr_glasso|pearson_corr_binary|pearson_corr|mutual_info")

def graphing(A, community_method=None, feats=True, plot=False, deg_trh=0, alpha=0e-0):
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
            "Laplacian Eigenvectors": features["Laplacian Eigenvectors"]
        }

        # Create DataFrames
        node_df = pd.DataFrame(feature_list)
        node_df['ROI'] = node_df['Node'].apply(lambda x: f'ROI_{x + 1}')    # Set nodes equal to ROIs in dataframe and remove nodes index
        node_df = node_df.drop(columns=['Node'])
        roi_col = node_df.pop('ROI')
        node_df.insert(0, 'ROI', roi_col)
        graph_df = pd.DataFrame([graph_features])

        return node_df, graph_df

def load_files(folder_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "abide")):
    """
    Load all .1D files from a folder where each file contains multiple time series (columns)

    Returns:
    - List of 2D numpy arrays (one per file)
    - List of filenames
    - List of institute names & subject ids
    - Dictionary with metadata about dimensions
    """
    # Get all .1D files sorted alphabetically
    file_list = sorted(glob.glob(os.path.join(folder_path,'*', '*.1D')))

    all_data = []
    subject_ids = []
    institute_names = []
    file_info = {
        'total_files': len(file_list),
        'timepoints_per_file': [],
        'series_per_file': []
    }

    for file_path in file_list:
        try:
            # Load all columns from the file
            data = np.loadtxt(file_path)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            all_data.append(data)
            filename = os.path.basename(file_path)

            # Extract institute name and subject ID from filename
            institute = filename.split('_005')[0]
            subject_id = '005' + filename.split('_005')[1].split('_')[0]

            institute_names.append(institute)
            subject_ids.append(subject_id)

            # Store metadata
            file_info['timepoints_per_file'].append(data.shape[0])
            file_info['series_per_file'].append(data.shape[1])

            print(f"Loaded {filename}: {data.shape[0]} timepoints × {data.shape[1]} series")

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            all_data.append(None)
            subject_ids.append(None)
            institute_names.append(None)

    return all_data, file_list, subject_ids, institute_names, file_info

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
    print("adj:\n",adj_matrix)
    print("corr:\n",corr_matrix)
    return adj_matrix, corr_matrix

def partial_corr(ts_data, method="ledoit", threshold=0.2):

    if method == 'glasso':
        variances = np.var(ts_data, axis=0)
        ts_data = ts_data[:, variances > 1e-8]
        ts_data = StandardScaler().fit_transform(ts_data)
        emp_cov = np.cov(ts_data)
        emp_cov += np.eye(emp_cov.shape[0]) * 1e-3
        inv_cov = GraphicalLassoCV(max_iter=5000).fit(emp_cov,ts_data).precision_
    else:
        inv_cov = LedoitWolf().fit(ts_data).precision_

    #inv_cov = np.linalg.pinv(cov)
    W = -inv_cov / np.sqrt(np.outer(np.diag(inv_cov), np.diag(inv_cov)))
    np.fill_diagonal(W, 0)

    # Apply thresholding to induce sparsity
    W[abs(W) < threshold] = 0
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

def Adj_heatmap(W):
    plt.figure(figsize=(8, 6))
    sns.heatmap(W,
                cmap='coolwarm',
                square=True)
    plt.xlabel("ROI")
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.show()

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

    # Functional connectivity matrix
    roi_pairs = list(combinations(aal_labels, 2))
    feature_names = [f"Corr_{r1}-{r2.split('_')[1]}" for r1, r2 in roi_pairs]

    #xf = x[:, np.var(x, axis=0) > 1e-8]
    fc_matrix = np.corrcoef(x)
    fc_matrix = np.nan_to_num(fc_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    triu_indices = np.triu_indices_from(fc_matrix, k=1)
    fc_vector = fc_matrix[triu_indices]
    df_fc = pd.DataFrame([fc_vector], columns=feature_names)

    return df, df_fc

def multiset_feats(data_list, method="partial_corr_LF"):
    """
    Loops over all data recursively to compute specific features and stores them in a dataframe indexed per individual (subject ID).

    - Input: set of 1D timeseries
    - Output: DataFrame indexed per individual with all relative features
    - Parameters: 'method' to choose graph inference method used

    """
    df_app = pd.DataFrame(columns=stat_feats(data_list[0])[0].columns) # Initialize dataframe
    df_global_list = []
    df_fc_list = []
    expanded_ids = []

    for data, sid in zip(data_list, load_files()[2]):
        if data is None:
            continue  # Skip if loading failed

        df_stat, df_fc = stat_feats(data) # Compute the statistical features
        df_graph, df_global = graphing(A= detect_inf_method(data, method=method)) # Compute the graphical features
        df_roi = pd.merge(df_stat, df_graph, on='ROI', how='left') # Merge both dataframes
        df_global_list.append(df_global)
        df_fc_list.append(df_fc)
        if df_app.empty:
            df_app = df_roi  # First iteration in case df_app=empty: set df_app = df
        else:
            df_app = pd.concat([df_app, df_roi], ignore_index=True)


        expanded_ids.extend([sid] * len(df_roi)) # Pad subject IDs with copies for all ROIs

    df_fc = pd.concat(df_fc_list, ignore_index=True)
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
    df = pd.concat([df_fc, df_global], axis=1)
    df_wide = pd.concat([df_wide[id_cols + features_grouped], df], axis=1)

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
# data_arrays, file_paths, subject_ids, institute_names, metadata = load_files() # represents format of load_files()

# represents multiset_feats() usage, add another index '[]' to load_files to select data amount
print("Output data:\n", multiset_feats(load_files()[0][:5]))