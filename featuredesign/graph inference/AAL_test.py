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

import seaborn as sns
from scipy.stats import pearsonr
from networkx.algorithms import community
from sklearn.covariance import GraphicalLasso, LedoitWolf
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import mutual_info_regression

def graphing(A, super=False, feats=False):
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
    low_degree_nodes = [node for node, degree in G.degree() if degree < 2]
    G.remove_nodes_from(low_degree_nodes)

    if super:
        communities = community.greedy_modularity_communities(G) # Combines nodes that are close together
        for i, c in enumerate(communities):
            print(f"Community {i}: {sorted(c)}")

        node_to_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
               node_to_community[node] = i

        # 4. Initialize a new graph for the supernodes
        G = nx.Graph()

        # Add one node per community
        for i in range(len(communities)):
            G.add_node(i)

        # 5. Add edges between supernodes (if any original edge connects nodes in different communities)
        for u, v in G.edges():
           cu = node_to_community[u]
           cv = node_to_community[v]
           if cu != cv:
               G.add_edge(cu, cv)

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


def build_correlation_graph(time_series_data, threshold=0.5, absolute_value=True):
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

    return adj_matrix, corr_matrix

def partial_correlation(ts_data, method = "ledoit", alpha=0.1):
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

def mutual_information(data):
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

    print(f"Shape used for feature extraction: {x.shape}")

    # Extract features for each time series
    feature_list = []
    for ts in x:
        features = {
            'mean': np.mean(ts),
            'std': np.std(ts),
            'SNR': np.divide(np.mean(x, axis=0), np.std(x, axis=0), where=np.std(x, axis=0) != 0, out=np.zeros_like(np.mean(x, axis=0)))
        }
        feature_list.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(feature_list)
    df.insert(0, 'ROI', [f'ROI_{i + 1}' for i in range(len(df))])
    df.to_csv('aal_feats.csv', index=False)
    return df

def multiset_feats(data_list, filenames, output_dir="feature_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    for data, path in zip(data_list, filenames):
        if data is None:
            continue  # Skip if loading failed

        df = stat_feats(data)

        # Use base filename without extension
        base_name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(output_dir, f"{base_name}_features.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aal_data_test")
data_arrays, file_paths, metadata = load_files(data_path)

print(data_arrays[0])

# Standardize data
scaler = StandardScaler()
time_series_data_scaled = scaler.fit_transform(data_arrays[0])

print(data_arrays[0])

x = data_arrays[0]
t = np.arange(len(x))

corr_matrix = np.corrcoef(x)
# Set a correlation threshold
threshold = 0.8

# Create an adjacency matrix (binary graph)
adj_matrix = (np.abs(corr_matrix) > threshold).astype(int)

# Remove self-loops
np.fill_diagonal(adj_matrix, 0)

# Compute functions
print(adj_matrix)
#S_parc = partial_correlation(x)
#S_mi = mutual_information(x)

# Plotting
plt.plot(t,x)
plt.xlabel("Time Index")
plt.ylabel("Magnitude")
plt.show()
#plot_connectivity(S_mi)
#graphing(adj_matrix)

from nilearn import datasets
# Fetch AAL atlas
#aal = datasets.fetch_atlas_aal()
#atlas_labels = aal['labels']  # e.g., ['Precentral_L', 'Frontal_Sup_L', ...]
# = aal['maps']       # nifti file with labeled ROIs

#stat_feats(x)

print(datasets)

multiset_feats(data_arrays, file_paths)