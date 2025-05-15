import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score
)

def get_specificity(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

def get_sensitivity(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn)

def get_metrics(ytrue, ypred, yprob):
    confmatr = confusion_matrix(ytrue, ypred)

    # # Plot confusion matrix
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(confmatr, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=['Control', 'Autism'], yticklabels=['Control', 'Autism'])
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix')
    # plt.tight_layout()
    # plt.show()

    # # Print classification report
    # print("Classification report:\n", classification_report(ytrue, ypred))

    # Metrics
    precision, recall, f1score, support = precision_recall_fscore_support(ytrue, ypred, zero_division=0)
    auroc = roc_auc_score(ytrue, yprob)
    spec = get_specificity(ytrue, ypred)
    sensi = get_sensitivity(ytrue, ypred)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1score,
        "support": support,
        "specificity": spec,
        "sensitivity": sensi,
        "auroc": auroc
    }

    return metrics
