import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearnextraction import *
from featureimportance import *

def plotConnectome(model, featnames):
    labels, maps, indices = extractaal()
    coords = plotting.find_parcellation_cut_coords(maps)
    idx2coord = {idx: coord for idx, coord in zip(indices, coords)}
    topfeats = getimportanceK(model, featnames)

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

    plotting.plot_connectome(adjacency_matrix=adjmatr, node_coords=nodecoords, title="Top 20 Most Important Connections", black_bg=True, colorbar=True)
    plt.show()

if __name__ == "__main__":
    # Quick testing
    # labels, maps, indices = extractaal()
    df = pd.read_csv('nilearnfeatscomb.csv.gz')
    X = df.iloc[:, 4:]
    y = df['DX_GROUP']
    if set(y.unique()) == {1, 2}:
        y = y.map({1: 1, 2: 0})
    Xtrain, Xtest, ytrain, ytest = performsplit(X, y)
    Xtrain, Xtest = normalizer(Xtrain, Xtest)
    lrmodel = applyLogR(Xtrain, ytrain)
    plotConnectome(lrmodel, featnames=X.columns)