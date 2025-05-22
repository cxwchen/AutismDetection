import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
from wisardpkg import ClusWisard
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

def bestSVM_GS(Xtrain, Xtest, ytrain, ytest, svcdefault):
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

    # model = gridsearch.best_estimator_
    # y_pred = model.predict(Xtest)

    # accuracy = accuracy_score(ytest, y_pred)
    # recall = recall_score(ytest, y_pred)
    # precision = precision_score(ytest, y_pred)
    # f1 = f1_score(ytest, y_pred)

    # print(f"Accuracy:  {accuracy:.4f}")
    # print(f"Recall:    {recall:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"F1 Score:  {f1:.4f}")

    # confmat = confusion_matrix(ytest, y_pred)
    # plt.figure(figsize=(8,6))
    # sns.heatmap(confmat, annot=True, fmt='d', cmap='flare')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix using GridSearchCV')
    # plt.show()

    return gridsearch.best_params_

def weighted_recall(y_true, y_pred):
    recall_pos = recall_score(y_true, y_pred, pos_label=1)
    recall_neg = recall_score(y_true, y_pred, pos_label=0)
    # Adjust weights as you prefer
    return 0.7 * recall_pos + 0.3 * recall_neg

def bestSVM_RS(Xtrain, Xtest, ytrain, ytest, svcdefault):
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

    # model = rsearch.best_estimator_
    # y_pred = model.predict(Xtest)

    # accuracy = accuracy_score(ytest, y_pred)
    # recall = recall_score(ytest, y_pred)
    # precision = precision_score(ytest, y_pred)
    # f1 = f1_score(ytest, y_pred)

    # print(f"Accuracy:  {accuracy:.4f}")
    # print(f"Recall:    {recall:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"F1 Score:  {f1:.4f}")

    # confmat = confusion_matrix(ytest, y_pred)
    # plt.figure(figsize=(8,6))
    # sns.heatmap(confmat, annot=True, fmt='d', cmap='flare')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix of SVM using RandomizedSearchCV')
    # plt.show(block=False)

    return rsearch.best_params_

def bestDT(Xtrain, Xtest, ytrain, ytest, dtdefault):
    param_grid = {
        'max_depth': [None] + list(np.arange(2, 20)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'criterion': ['gini', 'entropy', 'log_loss']
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

    # model = rsearch.best_estimator_
    # y_pred = model.predict(Xtest)

    # accuracy = accuracy_score(ytest, y_pred)
    # recall = recall_score(ytest, y_pred)
    # precision = precision_score(ytest, y_pred)
    # f1 = f1_score(ytest, y_pred)

    # print(f"Accuracy:  {accuracy:.4f}")
    # print(f"Recall:    {recall:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"F1 Score:  {f1:.4f}")

    # confmat = confusion_matrix(ytest, y_pred)
    # plt.figure(figsize=(8,6))
    # sns.heatmap(confmat, annot=True, fmt='d', cmap='flare')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix of Decision Tree using RandomizedSearchCV')
    # plt.show(block=False)

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

    # model = rsearch.best_estimator_
    # y_pred = model.predict(Xtest)

    # accuracy = accuracy_score(ytest, y_pred)
    # recall = recall_score(ytest, y_pred)
    # precision = precision_score(ytest, y_pred)
    # f1 = f1_score(ytest, y_pred)

    # print(f"Accuracy:  {accuracy:.4f}")
    # print(f"Recall:    {recall:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"F1 Score:  {f1:.4f}")

    # confmat = confusion_matrix(ytest, y_pred)
    # plt.figure(figsize=(8,6))
    # sns.heatmap(confmat, annot=True, fmt='d', cmap='flare')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix of Random Forest using RandomizedSearchCV')
    # plt.show(block=False)

    return rsearch.best_params_

def bestMLP(Xtrain, Xtest, ytrain, ytest, MLPdefault):
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': uniform(1e-5, 1e-2),
        'learning_rate_init': uniform(1e-4, 1e-1),
        'solver': ['adam', 'sgd']
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

    # model = rsearch.best_estimator_
    # y_pred = model.predict(Xtest)

    # accuracy = accuracy_score(ytest, y_pred)
    # recall = recall_score(ytest, y_pred)
    # precision = precision_score(ytest, y_pred)
    # f1 = f1_score(ytest, y_pred)

    # print(f"Accuracy:  {accuracy:.4f}")
    # print(f"Recall:    {recall:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"F1 Score:  {f1:.4f}")

    # confmat = confusion_matrix(ytest, y_pred)
    # plt.figure(figsize=(8,6))
    # sns.heatmap(confmat, annot=True, fmt='d', cmap='flare')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix of MLP using RandomizedSearchCV')
    # plt.show(block=False)

    return rsearch.best_params_
# def tune_minScore(Xtrain, Xtest, ytrain, ytest, addressSize, discriminatorLimit, minScore_values):
#     """
#     Hyperparameter tuning for ClusWisard minScore parameter.
    
#     Parameters
#     ----------
#     Xtrain : array-like
#         Training features
#     Xtest : array-like
#         Validation features
#     ytrain : array-like
#         Training labels
#     ytest : array-like
#         Validation labels
#     addressSize : int
#         The address size parameter for ClusWisard
#     discriminatorLimit : int, optional
#         Discriminator limit for ClusWisard (default 4)
#     minScore_values : list or np.array, optional
#         List of minScore values to try. If None, default linspace [0,1] with 20 points is used.
    
#     Returns
#     -------
#     best_minScore : float
#         The minScore value with the best validation accuracy
#     best_score : float
#         The best validation accuracy obtained
#     """

#     if minScore_values is None:
#         minScore_values = np.linspace(0, 1, 20)  # Default search range
    
#     if isinstance(ytrain, pd.Series):
#         ytrain = ytrain.to_numpy()
#     best_score = -np.inf
#     best_minScore = None
    
#     Xtrain_list = Xtrain.astype(int).tolist()

#     # Convert labels to list of strings (or keep as strings if already)
#     ytrain_str = [str(label) for label in ytrain]

#     for score in minScore_values:
#         model = ClusWisard(addressSize, float(score), discriminatorLimit, 1)
#         model.train(Xtrain_list, ytrain_str)

#         Xtest_list = Xtest.astype(int).tolist()

#         y_pred_str = model.classify(Xtest_list)
        
#         # Convert predictions back to int (if needed) for metrics
#         y_pred = [int(label) for label in y_pred_str]
        
#         acc = accuracy_score(ytest, y_pred)
        
#         print(f"minScore={score:.3f} -> Validation Accuracy: {acc:.4f}")
        
#         if acc > best_score:
#             best_score = acc
#             best_minScore = score
            
#     print(f"Best minScore: {best_minScore:.3f} with accuracy: {best_score:.4f}")
#     return best_minScore, best_score