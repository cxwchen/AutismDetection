# Classification
This folder contains all files used by the Classification subgroup.
It is split in 2 folders: `src` and `GUI`. 
## `src`
### `classifiers.py`
This file contains the implementation of all classifiers:
- `applyLogR`
- `applySVM`
- `applyDT`
- `applyRandForest`
- `applyMLP`
- `applyLDA`
- `applyKNN`
- `applyDummy`

The models are only fitted here. No predictions yet.
### `hyperparametertuning.py`
This file contains the implementation of our hyperparameter tuning for the models SVM, Decision Tree, and MLP.
### `classification.py`
This file performs the prediction and evaluation in `performCA()`.
### `performance.py`
This file contains all functions to calculate the performance metrics to evaluate the performance of our classifiers.

These functions are called by the `performCA()` function in `classification.py`.


