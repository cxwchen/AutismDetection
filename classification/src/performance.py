import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score
)

def get_specificity(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

def get_sensitivity(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn)

def plot_confusion_matrix(y_true, y_pred, model):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({model.__class__.__name__})')
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/conf_matrix_{model.__class__.__name__}.png', bbox_inches='tight')
    plt.show(block=False)  # Non-blocking
    # plt.pause(0.1)

def get_metrics(ytrue, ypred, yprob):

    # Print classification report
    print("Classification report:\n", classification_report(ytrue, ypred))

    # Metrics
    precision, recall, f1score, support = precision_recall_fscore_support(ytrue, ypred, zero_division=0)
    auroc = roc_auc_score(ytrue, yprob)
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
    classes = ['Control', 'Autism']
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

