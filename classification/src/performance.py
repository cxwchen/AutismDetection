import matplotlib.pyplot as plt
import os
import csv
import datetime
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score
)

# Generate timestamped CSV path at module load time
_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_CSV_PATH = f"results/eval_metrics_{_timestamp}.csv"

def get_specificity(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)
    if cm.shape != (2,2): #during CV evaluation per group 'SITE' a fold could end up with only one class.
        return float('nan')
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

def get_sensitivity(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)
    if cm.shape != (2, 2):
        return float('nan')
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn)

def plot_confusion_matrix(y_true, y_pred, model, fold=None, tag=""):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    title = f'Confusion Matrix ({model.__class__.__name__})'
    if tag:
        title += f' - {tag}'
    if fold is not None:
        title += f' - Fold {fold}'
    plt.title(title)
    os.makedirs('plots', exist_ok=True)
    filename = f'plots/conf_matrix_{model.__class__.__name__}'
    if tag:
        filename += f' - {tag}'
    if fold is not None:
        filename += f' - Fold {fold}'
    plt.savefig(f'{filename}.png', bbox_inches='tight')
    plt.close()
    # plt.show(block=False)  # Non-blocking
    # plt.pause(0.1)

def get_metrics(ytrue, ypred, yprob=None):
    """
    Compute common classification metrics.

    Parameters:
        ytrue : array-like
            True class labels
        ypred : array-like
            Predicted class labels
        yprob : array-like or None
            Probabilities or decision scores (optional for AUROC)

    Returns:
        metrics : dict
            Dictionary of computed metrics
    """
    print("Classification report:\n", classification_report(ytrue, ypred, zero_division=0))

    precision, recall, f1score, support = precision_recall_fscore_support(ytrue, ypred, zero_division=0)

    try:
        auroc = roc_auc_score(ytrue, yprob) if yprob is not None and len(np.unique(ytrue)) > 1 else float('nan')
    except:
        auroc = float('nan')  # AUROC can't be computed

    spec = get_specificity(ytrue, ypred)
    sensi = get_sensitivity(ytrue, ypred)
    accuracy = accuracy_score(ytrue, ypred)
    balanced_acc = balanced_accuracy_score(ytrue, ypred)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1score,
        "support": support,
        "specificity": spec,
        "sensitivity": sensi,
        "auroc": auroc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc
    }

    return metrics


def print_metrics(metrics, classifier_name="Classifier"):
    print(f"\n=== Metrics for {classifier_name} ===")

    precision = metrics["precision"]
    recall = metrics["recall"]
    f1_score = metrics["f1_score"]
    support = metrics["support"]
    specificity = metrics["specificity"]
    sensitivity = metrics["sensitivity"]
    auroc = metrics["auroc"]
    acc = metrics["accuracy"]
    balanced_acc = metrics["balanced_accuracy"]

    # Print per-class metrics
    print("\nPer-Class Metrics:")
    availclasses = len(precision)
    classes = ['Control', 'Autism'][:availclasses]
    for i, cls in enumerate(classes):
        print(f"  {cls}:")
        print(f"    Precision: {precision[i]:.3f}")
        print(f"    Recall:    {recall[i]:.3f}")
        print(f"    F1 Score:  {f1_score[i]:.3f}")
        print(f"    Support:   {support[i]}")

    # Print overall metrics
    print("\nOverall Metrics:")
    print(f"  Specificity:          {specificity:.3f}")
    print(f"  Sensitivity:          {sensitivity:.3f}")
    print(f"  AUROC:                {auroc:.3f}")
    print(f"  Accuracy:             {acc:.3f}")
    print(f"  Balanced Accuracy:    {balanced_acc:.3f}")
    print("=====================================")


def toCSV(csvpath, fold, classifierName, tag, group_col, group_name, metrics):
    os.makedirs(os.path.dirname(csvpath), exist_ok=True)

    rows = []
    for metric, value in metrics.items():
        if isinstance(value, (list, np.ndarray)):
            for i, v in enumerate(value):
                label = ['Control', 'Autism'][i] if len(value) == 2 else f"Class{i}"
                rows.append([fold, classifierName, tag, group_col, group_name, f"{group_name}_{label}", metric, v])
        else:
            rows.append([fold, classifierName, tag, group_col, group_name, metric, value])
    
    header = ['Fold', 'Classifier', 'Dataset', 'GroupType', 'GroupName', 'Metric', 'Value']
    writeHeader = not os.path.exists(csvpath)
    with open(csvpath, 'a', newline='') as f:
        writer = csv.writer(f) 
        if writeHeader:
            writer.writerow(header)
        writer.writerows(rows)

def perGroupEval(ytrue, ypred, yprob, meta, group_col, group_name, fold=None, classifier_name='Classifier', tag="", csv_path=DEFAULT_CSV_PATH):
    print(f"\n=== Metrics by {group_name} ===")
    groups = meta[group_col].unique()
    for group in groups:
        idx = meta[group_col] == group
        yt = np.array(ytrue)[idx]
        yp = np.array(ypred)[idx]
        yp_prob = np.array(yprob)[idx] if yprob is not None else None

        support = len(yt)
        unique, counts = np.unique(yt, return_counts=True)
        labelCts = {f"class_{int(u)}_count": int(c) for u, c in zip(unique, counts)}

        if len(np.unique(yt)) < 2:
            print(f"Skipping {group_name} = {group} --> only one class present.")
            metrics = {
                "precision": [float('nan')],
                "recall": [float('nan')],
                "f1_score": [float('nan')],
                "support": [support],
                "specificity": float('nan'),
                "sensitivity": float('nan'),
                "auroc": float('nan'),
                "accuracy": float('nan'),
                "balanced_accuracy": float('nan'),
                "test_size": support,
                **labelCts
            }
            toCSV(csv_path, fold, classifier_name, tag, group_name, group, metrics)
            continue
        
        print(f"\n {group_name}: {group}")
        metrics = get_metrics(yt, yp, yp_prob)
        metrics["test_size"] = support
        metrics.update(labelCts)
        print_metrics(metrics, f"{group_name}={group}")
        toCSV(csv_path, fold, classifier_name, tag, group_name, group, metrics)