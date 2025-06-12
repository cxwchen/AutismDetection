import os
import json
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearnextraction import *
from featureimportance import *

def plotConnectome(model, featnames, k, fold=None, tag="", timestamp="", save_feats=False):
    """
    -------------------------------------------------------------------------------------------
    This function visualises the most important brain connectivity features based on prediction
    -------------------------------------------------------------------------------------------
    This function builds an adjacency matrix weighted by feature importance and plots the 
    connectome graph using nilearn

    Parameters
    ----------
    model : object
        Fitted estimator
    
    featnames : array-like
        List, numpy array, or pandas Index of feature names (column labels)

    Returns
    -------
    None
    """

    labels, maps, indices = extractaal()
    coords = plotting.find_parcellation_cut_coords(maps)
    idx2coord = {idx: coord for idx, coord in zip(indices, coords)}
    topfeats = getimportanceK(model, featnames, k=k)

    if save_feats:
        os.makedirs(f"features/{timestamp}", exist_ok=True)
        save_path = f"features/{timestamp}/{model.__class__.__name__}"
        if tag:
            save_path += f"_{tag}"
        if fold is not None:
            save_path += f"_Fold{fold}"
        save_path += ".json"

        with open(save_path, "w") as f:
            json.dump(topfeats, f, indent=4)

    nodes = set()
    edgeinfo = []

    for fname, importance in topfeats:
        try:
            _, idx1, idx2 = fname.split('_')
            # idx1, idx2 = int(idx1), int(idx2)
            coord1 = idx2coord[idx1]
            coord2 = idx2coord[idx2]
            nodes.add(idx1)
            nodes.add(idx2)
            edgeinfo.append((idx1, idx2, abs(importance)))
        except KeyError:
            print(f"Warning: One of the indices {idx1} or {idx2} not in AAL atlas. Skipping")


    nodes = sorted(nodes)
    n = len(nodes)
    nodeidx = {node: i for i, node in enumerate(nodes)}

    adjmatr = np.zeros((n, n))
    nodecoords = []

    for node in nodes:
        nodecoords.append(idx2coord[node])

    for idx1, idx2, weight in edgeinfo:
        i, j = nodeidx[idx1], nodeidx[idx2]
        adjmatr[i,j] = weight
        adjmatr[j,i] = weight
    
    filename = f'plots/{timestamp}/{k}feats_{model.__class__.__name__}'
    if tag:
        filename += f' - {tag}'
    if fold is not None:
        filename += f' - Fold {fold}'

    plotting.plot_connectome(adjacency_matrix=adjmatr, node_coords=nodecoords, output_file=f'{filename}.png', title="Top 20 Most Important Connections", black_bg=False, colorbar=True)
    # plt.show()

def plotCustomConnectome(featlist, weight=1.0, tag="", filename="custom_top_feats", show=True):
    labels, maps, indices = extractaal()
    coords = plotting.find_parcellation_cut_coords(maps)
    idx2coord = {idx: coord for idx, coord in zip(indices, coords)}

    nodes = set()
    edgeinfo = []
    for fname in featlist:
        try: 
            _, idx1, idx2 = fname.split('_')
            coord1 = idx2coord[idx1]
            coord2 = idx2coord[idx2]
            nodes.add(idx1)
            nodes.add(idx2)
            edgeinfo.append((idx1, idx2, weight))
        except KeyError:
            print(f"Warning: one of the indices {idx1} or {idx2} not in AAL atlas. Skipping")

    nodes = sorted(nodes)
    n = len(nodes)
    nodeidx = {node: i for i, node in enumerate(nodes)}
    adjmatr = np.zeros((n, n))
    nodecoords = [idx2coord[node] for node in nodes]

    for idx1, idx2, w in edgeinfo:
        i, j = nodeidx[idx1], nodeidx[idx2]
        adjmatr[i,j] = w
        adjmatr[j, i] = w

    plotting.plot_connectome(adjacency_matrix=adjmatr, node_coords=nodecoords, output_file=f"{filename}.png", title="Top Stable Features")

def firsttest():
    df = pd.read_csv('nilearnfeatscomb.csv.gz')
    X = df.iloc[:, 4:]
    y = df['DX_GROUP']
    if set(y.unique()) == {1, 2}:
        y = y.map({1: 1, 2: 0})
    Xtrain, Xtest, ytrain, ytest = performsplit(X, y)
    Xtrain, Xtest = normalizer(Xtrain, Xtest)
    lrmodel = applyLogR(Xtrain, ytrain)
    plotConnectome(lrmodel, featnames=X.columns)

def plotConnectomeFromSaved(featfile, k, tag="", fold=None, timestamp="20250608_173602"):
    """
    -------------------------------------------------------------------------------------------
    This function visualises the most important brain connectivity features based on saved JSON
    -------------------------------------------------------------------------------------------
    This function loads feature importance data from a JSON file and builds an adjacency matrix
    weighted by signed feature importance, then plots the connectome graph using nilearn.

    Parameters
    ----------
    featfile : str
        Filename of the JSON file inside the folder features/{timestamp}/ (including extension)

    k : int
        Number of top features to plot

    tag : str, optional
        Additional tag in plot title (default is "")

    fold : int or None, optional
        Fold number for labeling the plot (default is None)

    timestamp : str, optional
        Folder name under features where JSON files are saved (default is "20250608_173602")

    Returns
    -------
    None
    """
    
    labels, maps, indices = extractaal()
    coords = plotting.find_parcellation_cut_coords(maps)
    idx2coord = {idx: coord for idx, coord in zip(indices, coords)}

    # Load saved top features
    filepath = os.path.join("features", timestamp, featfile)
    with open(filepath, "r") as f:
        topfeats = json.load(f)

    nodes = set()
    edgeinfo = []

    for fname, importance in topfeats:
        try:
            _, idx1, idx2 = fname.split('_')
            coord1 = idx2coord[idx1]
            coord2 = idx2coord[idx2]
            nodes.add(idx1)
            nodes.add(idx2)
            # Use signed importance, NOT abs()
            edgeinfo.append((idx1, idx2, importance))
        except KeyError:
            print(f"Warning: One of the indices {idx1} or {idx2} not in AAL atlas. Skipping")

    nodes = sorted(nodes)
    n = len(nodes)
    nodeidx = {node: i for i, node in enumerate(nodes)}

    adjmatr = np.zeros((n, n))
    nodecoords = []

    for node in nodes:
        nodecoords.append(idx2coord[node])

    for idx1, idx2, weight in edgeinfo:
        i, j = nodeidx[idx1], nodeidx[idx2]
        adjmatr[i, j] = weight
        adjmatr[j, i] = weight

    filename = f'plots/{timestamp}/{k}feats_from_saved'
    if tag:
        filename += f' - {tag}'
    if fold is not None:
        filename += f' - Fold {fold}'

    plotting.plot_connectome(adjacency_matrix=adjmatr,
                            node_coords=nodecoords,
                            output_file=f'{filename}.png',
                            title="Top 20 Most Important Connections (Signed)",
                            black_bg=False,
                            colorbar=True)

def plotallsaved(k=20, timestamp="20250608_173602"):
    folder = os.path.join("features", timestamp)
    json_files = glob.glob(os.path.join(folder, "*.json"))

    for json_file in json_files:
        filename = os.path.basename(json_file)
        # Try to parse fold number from filename if present
        fold = None
        if "Fold" in filename:
            try:
                fold_part = filename.split("Fold")[1]
                fold = int(fold_part.split(".")[0])
            except:
                pass

        # You can extract model name or tag from filename if you want
        tag = filename.replace(".json", "")
        print(f"Plotting {filename}...")
        plotConnectomeFromSaved(featfile=filename, k=k, tag=tag, fold=fold, timestamp=timestamp)


if __name__ == "__main__":
    # Quick testing
    # labels, maps, indices = extractaal()
    # firsttest()
    plotallsaved()