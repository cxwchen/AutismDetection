import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
from wisardpkg import ClusWisard


# traindata = load_features(file_path=...)
# testdata = load_features(file_path=...)
# feat_train = traindata.iloc(:, [...])
# ytrain = traindata.iloc(:, blablab)
# feat_test = testdata.iloc(:, [])
# ytest = testdata.iloc(:, jlfd)

param_grid = [
    {   'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}, #Here I didn't use gamma, because gamma is not used for the linear kernel
    {   'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]},
    {   'kernel': ['poly'], 'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'degree': [2, 3, 4],
        'coef0': [0.0, 0.1, 0.5, 1.0]},
    {   'kernel': ['sigmoid'], 'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'coef0': [0.0, 0.1, 0.5, 1.0]}
]

def defaultSVM(Xtrain, Xtest, ytrain, ytest):
    svc_default = SVC()
    # Train the model
    svc_default.fit(Xtrain, ytrain)

    # Make predictions on the test data
    y_pred = svc_default.predict(Xtest)

    # Calculate performance metrics
    accuracy = accuracy_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)

    # Print performance metrics
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")

    # Generate and plot confusion matrix
    conf_matrix = confusion_matrix(ytest, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def bestSVM_GS(Xtrain, Xtest, ytrain, ytest, paramgrid, svcdefault):
    gridsearch = GridSearchCV(svcdefault, paramgrid)
    gridsearch.fit(Xtrain, ytrain)

    model = gridsearch.best_estimator_
    y_pred = model.predict(Xtest)

    accuracy = accuracy_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    confmat = confusion_matrix(ytest, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(confmat, annot=True, fmt='d', cmap='flare')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix using GridSearchCV')
    plt.show()

    return gridsearch.best_params_

def bestSVM_RS(Xtrain, Xtest, ytrain, ytest, paramgrid, svcdefault):
    rsearch = RandomizedSearchCV(svcdefault, paramgrid, cv=5, random_state=19, n_iter=50)
    rsearch.fit(Xtrain, ytrain)

    model = rsearch.best_estimator_
    y_pred = model.predict(Xtest)

    accuracy = accuracy_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    confmat = confusion_matrix(ytest, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(confmat, annot=True, fmt='d', cmap='flare')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix using RandomizedSearchCV')
    plt.show()

    return rsearch.best_params_


def tune_minScore(Xtrain, Xtest, ytrain, ytest, addressSize, discriminatorLimit, minScore_values):
    """
    Hyperparameter tuning for ClusWisard minScore parameter.
    
    Parameters
    ----------
    Xtrain : array-like
        Training features
    Xtest : array-like
        Validation features
    ytrain : array-like
        Training labels
    ytest : array-like
        Validation labels
    addressSize : int
        The address size parameter for ClusWisard
    discriminatorLimit : int, optional
        Discriminator limit for ClusWisard (default 4)
    minScore_values : list or np.array, optional
        List of minScore values to try. If None, default linspace [0,1] with 20 points is used.
    
    Returns
    -------
    best_minScore : float
        The minScore value with the best validation accuracy
    best_score : float
        The best validation accuracy obtained
    """

    if minScore_values is None:
        minScore_values = np.linspace(0, 1, 20)  # Default search range
    
    best_score = -np.inf
    best_minScore = None
    
    for score in minScore_values:
        model = ClusWisard(addressSize, minScore=score, discriminatorLimit=discriminatorLimit)
        model.fit(Xtrain, ytrain)
        
        y_pred = model.predict(Xtest)
        acc = accuracy_score(ytest, y_pred)
        
        print(f"minScore={score:.3f} -> Validation Accuracy: {acc:.4f}")
        
        if acc > best_score:
            best_score = acc
            best_minScore = score
            
    print(f"Best minScore: {best_minScore:.3f} with accuracy: {best_score:.4f}")
    return best_minScore, best_score