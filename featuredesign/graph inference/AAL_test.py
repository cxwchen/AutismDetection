from nilearn import image, plotting
from nilearn.glm import first_level

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob
import nibabel as nib
import scipy as sp
import pandas as pd
import re
from Normalized_laplacian import learn_normalized_laplacian
import seaborn as sns
from scipy.stats import pearsonr
from networkx.algorithms import community
from sklearn.covariance import GraphicalLasso, LedoitWolf
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import mutual_info_regression

def graphing(A, super=False, feats=False, deg_trh=0):
    """
    Function converting Adjacency matrix to a Graph
    It has an input parameter 'super' to select supernodes
    Note that A is a binary matrix and a type of shift operator(S)
    Therefore A can be modeled as some image data consisting of a pixel grid
    """
    G = nx.from_numpy_array(A)

    # Extract edge weights and normalize for linewidth
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # Normalize weights to range [1, 5] for opacities
    if len(weights) > 0:
        min_w = min(weights)
        max_w = max(weights)
        if max_w != min_w:
            opacities = [0.2 + 0.8 * (w - min_w) / (max_w - min_w) for w in weights]
        else:
            opacities = [0.6] * len(weights)  # All equal if no variation
    else:
        opacities = []

    # Remove nodes with degree < 2
    low_degree_nodes = [node for node, degree in G.degree() if degree < deg_trh]
    G.remove_nodes_from(low_degree_nodes)

    if super:
        # Detect communities
        #communities = list(community.greedy_modularity_communities(G))
        communities = nx_comm.louvain_communities(G, resolution=0.8)
        print(f"Found {len(communities)} communities: {communities}")  # Debug
        communities = [comm for comm in communities if len(comm) >= 2]  # Keep only size >= 2
        print(f"Filtered communities: {communities}")

        # Map original nodes to their community IDs
        node_to_community = {node: i for i, comm in enumerate(communities) for node in comm}

        # Create supernode graph
        superG = nx.Graph()
        superG.add_nodes_from(range(len(communities)))  # One node per community

        # Track edges between communities
        inter_community_edges = set()

        # Check ALL original edges for inter-community connections
        for u, v in nx.from_numpy_array(A).edges():  # Use original adjacency matrix
            cu = node_to_community.get(u, -1)
            cv = node_to_community.get(v, -1)
            if cu != cv and cu != -1 and cv != -1:
                # Add edge between communities (sorted to avoid duplicates like (1,0) vs (0,1))
                edge = tuple(sorted((cu, cv)))
                inter_community_edges.add(edge)

        # Add edges to superG
        superG.add_edges_from(inter_community_edges)
        print(f"Supernode edges: {superG.edges()}")

        # Update references for visualization
        G = superG
        edges = list(superG.edges())
        opacities = [0.6] * len(edges)  # Uniform opacity for superedges

    # Create label dictionary: show node + 1
    labels = {node: node + 1 for node in G.nodes()}
    # 6. Visualize the coarsened graph
    pos = nx.spring_layout(G, k=0.5, seed=42, iterations=50)
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, labels=labels,
                    with_labels=True,
                    node_color='skyblue',
                    node_size=300,
                    font_weight='bold')
    if edges:
        nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=opacities,)
    plt.show()

    # 7. Compute Graph features
    if feats:
        degree_centrality = nx.degree_centrality(G)
        print("Degree Centrality:", degree_centrality)
        closeness = nx.closeness_centrality(G)
        print("Closeness Centrality:", closeness)
        eigenvector = nx.eigenvector_centrality(G)
        print("Eigenvector Centrality:", eigenvector)
        clustering = nx.clustering(G)
        print("Clustering Coefficient:", clustering)
        shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
        print("Shortest Path from Node 0:", shortest_paths[0])
        print("Average Clustering:", nx.average_clustering(G))
        print("Diameter:", nx.diameter(G))  # Longest shortest path
        edge_betweenness = nx.edge_betweenness_centrality(G)
        print("Edge Betweenness:", edge_betweenness)

def load_files(folder_path):
    """
    Load all .1D files from a folder where each file contains multiple time series (columns)

    Returns:
    - List of 2D numpy arrays (one per file)
    - List of filenames
    - Dictionary with metadata about dimensions
    """
    # Get all .1D files sorted alphabetically
    file_list = sorted(glob.glob(os.path.join(folder_path, '*.1D')))

    all_data = []
    file_info = {
        'total_files': len(file_list),
        'timepoints_per_file': [],
        'series_per_file': []
    }

    for file_path in file_list:
        try:
            # Load all columns from the file
            data = np.loadtxt(file_path)

            all_data.append(data)

            # Store metadata
            file_info['timepoints_per_file'].append(data.shape[0])
            file_info['series_per_file'].append(data.shape[1])

            print(f"Loaded {os.path.basename(file_path)}: {data.shape[0]} timepoints × {data.shape[1]} series")

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            all_data.append(None)

    return all_data, file_list, file_info


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
        for j in range(i, n_series):
            if i == j:
                corr = 1.0  # correlation with self is 1
            else:
                corr, _ = pearsonr(time_series_data[:, i], time_series_data[:, j])

            if absolute_value:
                corr = abs(corr)

            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

            # Create edge if correlation exceeds threshold
            if corr > threshold:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    # Remove self-loops
    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix, corr_matrix

def partial_corr(ts_data, method = "ledoit", alpha=0.1):
    """
    alpha: regularization parameter (smaller = denser connections)
    """
    try:
        if method == 'glasso':
            cov = GraphicalLasso(alpha=alpha, max_iter=5000).fit(ts_data).covariance_
        else:  # Fallback to Ledoit-Wolf
            cov = LedoitWolf().fit(ts_data)
    finally:
        inv_cov = np.linalg.pinv(cov)
        W = -inv_cov / np.sqrt(np.outer(np.diag(inv_cov), np.diag(inv_cov)))
        np.fill_diagonal(W, 0)
    return W

def mutual_info(data):
    N = data.shape[1]
    mi_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            # sklearn's MI expects 2D input for X and 1D for y
            mi = mutual_info_regression(data[:, i].reshape(-1, 1), data[:, j])[0]
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    return mi_matrix

def plot_connectivity(W):
    plt.figure(figsize=(8, 6))
    sns.heatmap(W,
                cmap='coolwarm',
                square=True)
    plt.xlabel("ROI")
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.show()

def stat_feats(x, n_rois = 116):
    """
    Compute classic statistical features based on some time series data and stores it to a csv file.

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
            'mean': np.mean(ts),
            'std': np.std(ts),
            'SNR': np.average(np.divide(np.mean(x, axis=0), np.std(x, axis=0), where=np.std(x, axis=0) != 0, out=np.zeros_like(np.mean(x, axis=0))))
        }
        feature_list.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(feature_list)
    df.insert(0, 'ROI', [f'ROI_{i + 1}' for i in range(len(df))])
    #df.to_csv('aal_feats.csv', index=False)
    #print(df)
    return df

def multiset_feats(data_list, filenames, output_dir="feature_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    df_app = pd.DataFrame(columns=stat_feats(data_list[0]).columns) # Initialize dataframe

    for data, path in zip(data_list, filenames):
        if data is None:
            continue  # Skip if loading failed

        df = stat_feats(data)
        if df_app.empty:
            df_app = df  # First iteration: set df_app = df
        else:
            df_app = pd.concat([df_app, df], ignore_index=True)
        print(df)
        print(df_app)

    df_app['subject_id'] = df_app.index // 116
    df_wide = df_app.pivot(index='subject_id', columns='ROI', values=['mean', 'std', 'SNR'])

    # Flatten column names
    df_wide.columns = [f"{stat}_{roi}" for stat, roi in df_wide.columns]
    df_wide = df_wide.reset_index()

    # Function to extract ROI number for sorting
    def extract_roi_num(col):
        match = re.search(r'_(\d+)$', col)
        return int(match.group(1)) if match else -1

    # Sort columns: keep subject_id first, then group by ROI
    cols = df_wide.columns.tolist()
    subject_col = ['subject_id']
    feature_cols = [col for col in cols if col != 'subject_id']

    # Group by ROI number
    features_grouped = sorted(feature_cols, key=lambda x: (extract_roi_num(x), x.split('_')[0]))

    # Reorder columns
    ordered_cols = subject_col + features_grouped
    df_wide = df_wide[ordered_cols]

    return df_wide.head()

folder_path = r"C:\Users\Jochem\Documents\GitHub\AutismDetection\abide\female-cpac-filtnoglobal-aal" # Enter your local ABIDE dataset path
data_arrays, file_paths, metadata = load_files(folder_path)

#output = multiset_feats(data_arrays, file_paths)
#stat_feats(data_arrays[0])

Laplacian = learn_normalized_laplacian(data_arrays[0], epsilon=1, alpha=0.1)
print(Laplacian)
graphing(Laplacian, feats=True)