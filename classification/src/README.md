# `src`
This folder contains all files used to implement and evaluate our classifiers for our thesis.

Below we give a short explanation of each file.
## `classifiers.py`
This file contains the implementation of all classifiers:
- Logistic Regression: `applyLogR`
- Support Vector Machine: `applySVM`
- Decision Tree: `applyDT`
- Random Forest: `applyRandForest`
- Multi-Layer Perceptron: `applyMLP`
- Linear Discriminant Analysis: `applyLDA`
- K-Nearest Neighbors: `applyKNN`
- Dummy (baseline): `applyDummy`

The models are only fitted here. No predictions yet.
## `hyperparametertuning.py`
This file contains the implementation of our hyperparameter tuning for the models SVM, Decision Tree, and MLP.
## `classification.py`
This file performs the prediction and evaluation in `performCA()`.
## `performance.py`
This file contains all functions to calculate the performance metrics to evaluate the performance
of our classifiers.

These functions are called by the `performCA()` function in `classification.py`.
## `loaddata.py`
This file contains functions `load_data()` to call the feature extraction implementation by the Feature Design
subgroup and `add_phenotypic_info()` to append the phenotypic info.

❗These functions are not used anymore as the Feature Design subgroup delivered the final features as
dataframes in the end.

`performsplit()`: an old function to perform a quick train-test split. This function is again not used in our research paper, as we
used stratified K-Fold and Leave-One-Group-Out instead.

`normalizer()`: this functions performs the (column-wise) standardisation to standardise the feature vectors.

`applyHarmo()`: this function is responsible for the harmonisation of the Pearson correlation features before
classification is performed.

## `nilearnextraction.py`
This file contains functions `nilearnextract()` to fetch the ABIDE I dataset, preprocess it with the CPAC pipeline,
and compute the Pearson correlation features and `extractaal()` to fetch the AAL atlas.

## `nilearndetection.py`
This file contains all functions that perform stratified K-Fold cross-validation
and Leave-One-Group-Out cross-validation used to evaluate the Pearson correlation
features. It calls `performCA()` from `classification.py` for the evaluations and the functions 
in `hyperparametertuning.py` for the hyperparameter tuning inside each fold.

Everything is logged (can be seen in the folder `logs`) and performance metrics are written to a 
CSV file in the folder `results`.

## `graphclassification.py`
This file contains the functions that perform stratified K-Fold cross-validation
and Leave-One-Group-Out cross-validation used to evaluate the graph features by the Feature Design subgroup.

## `featureimportance.py`
Contains the function `getimportanceK()` to return the K most important features and their 
importance scores. In our paper we set K to 20.

## `visualise.py` 
This file is responsible for the visualisation of the most important features. We use a
colorblind friendly colormap.
