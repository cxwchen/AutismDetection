import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import randint, uniform

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

def bestSVM_GS(Xtrain, Xtest, ytrain, ytest, svcdefault=SVC()):
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
    gridsearch = GridSearchCV(svcdefault, param_grid)
    gridsearch.fit(Xtrain, ytrain)

    return gridsearch.best_params_

def weighted_recall(y_true, y_pred):
    recall_pos = recall_score(y_true, y_pred, pos_label=1)
    recall_neg = recall_score(y_true, y_pred, pos_label=0)
    # Adjust weights as you prefer
    return 0.7 * recall_pos + 0.3 * recall_neg

def bestSVM_RS(Xtrain, Xtest, ytrain, ytest, svcdefault=SVC()):
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
    weighted_recall_scorer = make_scorer(weighted_recall)
    rsearch = RandomizedSearchCV(svcdefault, param_grid, scoring=weighted_recall_scorer, cv=5, random_state=19, n_iter=50, verbose=2)
    print("Starting RandomizedSearchCV fitting...")
    rsearch.fit(Xtrain, ytrain)
    print("RandomizedSearchCV fitting completed.")
    print(f"Best Parameters Found: {rsearch.best_params_}")

    return rsearch.best_params_

def bestDT(Xtrain, Xtest, ytrain, ytest, dtdefault):
    param_grid = {
        'max_depth': [None] + list(np.arange(2, 20)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'criterion': ['gini', 'entropy']
    }

    weighted_recall_scorer = make_scorer(weighted_recall)

    rsearch = RandomizedSearchCV(
        dtdefault,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        scoring=weighted_recall_scorer,
        random_state=19,
        n_jobs=-1,
        verbose=2
    )
    
    print("Starting RandomizedSearchCV fitting...")
    rsearch.fit(Xtrain, ytrain)
    print("RandomizedSearchCV fitting completed.")
    print(f"Best Parameters Found: {rsearch.best_params_}")

    return rsearch.best_params_

def bestRF(Xtrain, Xtest, ytrain, ytest, rfdefault):
    param_grid = {
        'n_estimators': randint(50, 300),
        'max_depth': [None] + list(np.arange(5, 31, 5)),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    weighted_recall_scorer = make_scorer(weighted_recall)

    rsearch = RandomizedSearchCV(
        rfdefault,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        scoring=weighted_recall_scorer,
        random_state=19,
        n_jobs=-1,
        verbose=2
    )
    
    print("Starting RandomizedSearchCV fitting...")
    rsearch.fit(Xtrain, ytrain)
    print("RandomizedSearchCV fitting completed.")
    print(f"Best Parameters Found: {rsearch.best_params_}")

    return rsearch.best_params_

def bestMLP(Xtrain, Xtest, ytrain, ytest, MLPdefault):
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': uniform(1e-5, 1e-2),
        'learning_rate_init': uniform(1e-4, 1e-1),
        'solver': ['adam', 'sgd'],
        'early_stopping': [True, False]
    }

    weighted_recall_scorer = make_scorer(weighted_recall)

    rsearch = RandomizedSearchCV(
        MLPdefault,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        scoring=weighted_recall_scorer,
        random_state=19,
        n_jobs=-1,
        verbose=2
    )
    
    print("Starting RandomizedSearchCV fitting...")
    rsearch.fit(Xtrain, ytrain)
    print("RandomizedSearchCV fitting completed.")
    print(f"Best Parameters Found: {rsearch.best_params_}")

    return rsearch.best_params_