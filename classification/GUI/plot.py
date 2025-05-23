# -*- coding: utf-8 -*-
"""
Created on Thu May 22 22:46:42 2025

@author: kakis
"""
import numpy as np
from nilearn import datasets, plotting
from nilearn.image import math_img, resample_to_img, get_data

# haxby dataset to have EPI images and masks
haxby_dataset = datasets.fetch_haxby()

# print basic information on the dataset
print(
    f"First subject anatomical nifti image (3D) is at: {haxby_dataset.anat[0]}"
)
print(
    f"First subject functional nifti image (4D) is at: {haxby_dataset.func[0]}"
)


def vis(haxby_dataset,rois):
    haxby_anat_filename = haxby_dataset.anat[0]  # First subject's anatomical scan
    # haxby_mask_filename = haxby_dataset.mask_vt[0]
    # haxby_func_filename = haxby_dataset.func[0]
    
    # Load AAL Atlas
    aal = datasets.fetch_atlas_aal()
    aal_map = aal.maps  # 3D NIfTI image with integer labels per ROI
    aal_labels = aal.labels  # List of ROI names
    aal_resampled = resample_to_img(aal_map, haxby_anat_filename, interpolation='nearest')
    resampled_data = get_data(aal_resampled)
    unique_labels = np.unique(resampled_data)
    
    important_roi_indices = [int(unique_labels[roi - 2000]) for roi in rois]
    
    # === plot_roi funtion:
    
    # Expression for math_img like: (img == 5) + (img == 10) + (img == 15)
    expr = " + ".join([f"(img == {roi})" for roi in important_roi_indices])
    roi_mask = math_img(expr, img=aal_resampled)
    
    # Plot the ROI mask on the anatomical image
    plotting.plot_roi(roi_mask, bg_img=haxby_anat_filename,
                      title=" + ".join([f"{aal_labels[idx - 2001]}" for idx in important_roi_indices]),
                       alpha=1.0)
    
    # === print_roi function:
    for idx in important_roi_indices:    
        print(f"{idx}: {aal_labels[idx - 2001]}")
        
    roi_names = [aal_labels[roi - 2000] for roi in rois]
    return roi_names
    
vis(haxby_dataset,[2001,2002])       
 
