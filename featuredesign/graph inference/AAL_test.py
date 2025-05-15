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
import seaborn as sns
import networkx.algorithms.community as nx_comm

from scipy.stats import pearsonr,skew, kurtosis
from networkx.algorithms import community
from sklearn.covariance import GraphicalLasso, LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from Normalized_laplacian import learn_normalized_laplacian


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

def graphing(A, super=False, feats=False, plot=True, deg_trh=0, alpha=0e-0):
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
        min_w = min(weights_filt)
        max_w = max(weights_filt)

        # Handle case where all weights are equal
        if max_w == min_w:
            opacities = [0.5 for _ in weights_filt]  # Set uniform opacity
        else:
            opacities = [0.2 + 0.8 * (w - min_w) / (max_w - min_w) for w in weights_filt]
    else:
        opacities = []

    # Remove nodes with degree < degree threshold
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

    if plot:
        # Create label dictionary: show node + 1
        labels = {node: node + 1 for node in G.nodes()}
        # 6. Visualize the coarsened graph
        pos = nx.spring_layout(G, k=0.5, seed=42, iterations=50)
        plt.figure(figsize=(6, 6))
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=100)
        nx.draw_networkx_labels(G, pos, labels=labels, font_weight='bold')
        if edges_filt:
            nx.draw_networkx_edges(G, pos, edgelist=edges_filt, alpha=opacities,)
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
            "Diameter": features["Diameter"]
        }

        # Create DataFrames
        node_df = pd.DataFrame(feature_list)
        node_df['ROI'] = node_df['Node'].apply(lambda x: f'ROI_{x + 1}')    # Set nodes equal to ROIs in dataframe and remove nodes index
        node_df = node_df.drop(columns=['Node'])
        roi_col = node_df.pop('ROI')
        node_df.insert(0, 'ROI', roi_col)
        graph_df = pd.DataFrame([graph_features])

        print(node_df)
        print(graph_df)

        return node_df, graph_df

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

            corr_matrix[i, j] = corr_matrix[j, i] = corr

            # Threshold for adjacency (skip diagonal)
            if i != j and corr > threshold:
                adj_matrix[i, j] = adj_matrix[j, i] = 1

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

def Adj_heatmap(W):
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
            'mean': np.mean(ts),
            'std': np.std(ts),
            'SNR': np.average(np.divide(np.mean(x, axis=0), np.std(x, axis=0), where=np.std(x, axis=0) != 0, out=np.zeros_like(np.mean(x, axis=0)))),
            'Skewness': skew(ts),
            'Kurtosis': kurtosis(ts)
        }
        feature_list.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(feature_list)
    df.insert(0, 'ROI', [f'ROI_{i + 1}' for i in range(len(df))])
    #df.to_csv('aal_feats.csv', index=False)
    #print(df)
    return df

def multiset_feats(data_list, filenames, output_dir="feature_outputs"):
    """
    Loops over all data recursively to compute specific features and stores them in a dataframe indexed per individual (subject ID).

    """
    os.makedirs(output_dir, exist_ok=True)
    df_app = pd.DataFrame(columns=stat_feats(data_list[0]).columns) # Initialize dataframe

    for data, path in zip(data_list, filenames):
        if data is None:
            continue  # Skip if loading failed

        df_stat = stat_feats(data) # Compute the statistical features
        df_graph,df_ex = graphing(A= pearson_corr(data)[0], feats=True, plot=False) # Compute the graphical features using pearson correlation adjacency matrix
        df_conc = pd.merge(df_stat, df_graph, on='ROI', how='left') # Merge both region of interests of statistical and graph feature dataframes.
        if df_app.empty:
            df_app = df_conc  # First iteration in case df_app=empty: set df_app = df
        else:
            df_app = pd.concat([df_app, df_conc], ignore_index=True)

    print("The appended dataframe:\n", df_app) # Appended dataframe for all inviduals in the dataset

    df_app['subject_id'] = df_app.index // 116
    df_wide = df_app.pivot(index='subject_id', columns='ROI', values=df_app.columns.difference(['ROI', 'subject_id'])) # Automatically pivots all feature columns

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

    return df_wide



#-------{Main for testing}-------#
folder_path = r"C:\Users\Jochem\Documents\GitHub\AutismDetection\abide\female-cpac-filtnoglobal-aal" # Enter your local ABIDE dataset path
data_arrays, file_paths, metadata = load_files(folder_path)

#stat_feats(data_arrays[0])
#print(output)
#print(data_arrays[0])
#Laplacian = learn_normalized_laplacian(data_arrays[0], epsilon=5e-1, alpha=0.1)
#print(Laplacian.shape)
#S = mutual_info(data_arrays[2])
#A, C = pearson_corr(data_arrays[2])
#print(C)
#Adj_heatmap(C)
#graphing(Laplacian, alpha=0.1)

output = multiset_feats(data_arrays[:5], file_paths)
print(output)
print(len(data_arrays))
