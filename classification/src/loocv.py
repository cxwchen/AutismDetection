from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from classifiers import *
from loaddata import normalizer
from hyperparametertuning import *
from sklearn.model_selection import LeaveOneOut
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    f1_score, balanced_accuracy_score, roc_auc_score
)
import numpy as np

comb_df = pd.read_csv("nilearnfeatscomb.csv.gz").sample(frac=1, random_state=42).reset_index(drop=True)
comb_df.rename(columns={"AGE_AT_SCAN": "AGE"}, inplace=True)
female_df = comb_df[comb_df['SEX'] == 2].sample(frac=1, random_state=42).reset_index(drop=True)
NYUfem_df = female_df[female_df["SITE_ID"]=="NYU"].reset_index(drop=True)

def runLOOCV(df, func):
    X = df.iloc[:, 4:].to_numpy()
    y = df['DX_GROUP'].to_numpy()

    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    y_prob = []  # for AUROC

    for train_index, test_index in loo.split(X):
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]

        imputer = SimpleImputer(strategy='mean')
        Xtrain = imputer.fit_transform(Xtrain)
        Xtest = imputer.transform(Xtest)
        Xtrain, Xtest = normalizer(Xtrain, Xtest)

        if func == applySVM:
            params = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, SVC())
        elif func == applyDT:
            params = bestDT(Xtrain, Xtest, ytrain, ytest, DecisionTreeClassifier())
        elif func == applyMLP:
            params = bestMLP(Xtrain, Xtest, ytrain, ytest, MLPClassifier())
    # rfparams = bestRF(Xtrain, Xtest, ytrain, ytest, RandomForestClassifier())
        else:
            params = None

        model = func(Xtrain, ytrain, params=params)  # your custom function
        pred = model.predict(Xtest)
        y_true.append(ytest[0])
        y_pred.append(pred[0])

        # Get predicted probability for AUROC (for positive class only)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(Xtest)[0][1]
            y_prob.append(prob)
        else:
            y_prob.append(0.5)  # fallback if prob not supported

    # Convert lists to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)   # sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    # Specificity (manual from confusion matrix)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Print results
    print(f"RESULTS FOR {func}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall (Sensitivity): {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"AUROC: {auc:.3f}")

if __name__=="__main__":
    # runLOOCV(NYUfem_df, applyLogR)
    runLOOCV(NYUfem_df, applySVM)
    # runLOOCV(NYUfem_df, applyDT)
    # runLOOCV(NYUfem_df, applyRandForest)
    # runLOOCV(NYUfem_df, applyMLP)
    # runLOOCV(NYUfem_df, applyLDA)
    # runLOOCV(NYUfem_df, applyKNN)