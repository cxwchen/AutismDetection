import os
from nilearn import datasets, plotting
from nilearn.datasets import fetch_abide_pcp, fetch_atlas_aal
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import seaborn as sns

# Function to process one atlas
def nilearnextract():
    """
    ----------------------------------------------------------------------
    This function extracts the Pearson correlation between ROIs using Nilearn
    ----------------------------------------------------------------------

    Returns
    -------
    df : DataFrame
        A DataFrame containing both features as well as the phenotypic info used for evaluation
    
    labels : list of str
        list of the names of the regions. Version: SPM12
    
    maps : NiftiImage
        path to nifti file containing the regions.
    
    indices : list of str
        indices mapping 'labels' to values in the 'maps' image
    """

    data = fetch_abide_pcp(pipeline='cpac', band_pass_filtering=True, global_signal_regression=False, derivatives='rois_aal', quality_checked=True)
    phenotypic = data.phenotypic
    aal = fetch_atlas_aal(version='SPM12')
    labels = aal.labels
    maps = aal.maps
    indices = aal.indices
    n_rois = len(labels)
    roipairs = []
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            roipairs.append(f"fc_{indices[i]}_{indices[j]}")

    X = data["rois_aal"]
    fullcorr = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
    feats = fullcorr.fit_transform(X)
    feats = pd.DataFrame(feats, columns=roipairs)
    demographics = phenotypic[["AGE_AT_SCAN", "SEX", "SITE_ID", "DX_GROUP"]].copy()
    demographics["DX_GROUP"] = demographics["DX_GROUP"].map({1: 1, 2: 0})
    df = pd.concat([demographics.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    return df, labels, maps, indices

def extractaal():
    """
    ------------------------------------
    This function fetches the AAL atlas
    ------------------------------------

    Returns
    -------
    labels : list of str
        list of the names of the regions. Version: SPM12
    
    maps : NiftiImage
        path to nifti file containing the regions.
    
    indices : list of str
        indices mapping 'labels' to values in the 'maps' image
    """

    aal = fetch_atlas_aal(version='SPM12')
    labels = aal.labels
    maps = aal.maps
    indices = aal.indices
    return labels, maps, indices


if __name__ == "__main__":
    data, labels, maps, indices = nilearnextract()
    data.to_csv('nilearnfeatscomb.csv.gz', index=False, compression='gzip')