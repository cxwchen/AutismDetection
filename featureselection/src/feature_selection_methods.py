import numpy as np
from HSIC import hsic_gam
from sklearn.linear_model import Lasso, LassoLars
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def greedy_hsic_lasso(X, y, k, redundancy_penalty=0.5):
   # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_reshaped = np.array(y).reshape(-1,1)

    n_features = X.shape[1]
    selected = []
    remaining = list(range(n_features))

    #Compute the HSIC values relevance
    relevance = []
    for i in range(n_features):
        X_feat = X_scaled[:,i].reshape(-1,1)
        hsic_value, _ = hsic_gam(X_feat, y_reshaped, alph=0.5)
        relevance.append(hsic_value)
    relevance = np.array(relevance)

    #Greedy feature selection
    for _ in range(k):
        best_score = -np.inf
        best_feature = None

        for i in remaining:
            redundancy = 0
            for j in selected:
                non_score, _ = hsic_gam(X_scaled[:,i].reshape(-1,1), X_scaled[:,j].reshape(-1,1), alph=0.5)
                redundancy += non_score
            score = relevance[i] - redundancy_penalty * redundancy

            if score > best_score:
                best_score = score
                best_feature = i

        selected.append(best_feature)
        remaining.remove(best_feature)
        print(f"Selected feature {best_feature} with score {best_score:.4f}")

    return selected

def lars_lasso(X, y, alpha=1.0, max_iter=100):
   # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_reshaped = np.array(y).reshape(-1,1)
        
    # Perform Lasso regression to select features based on HSIC values
    lasso = LassoLars(alpha=alpha, max_iter=max_iter, eps=1e-6)
    lasso.fit(X_scaled, y)
    
    # Select the features with non-zero coefficients
    selected_features = np.where(lasso.coef_ != 0)[0]
    
    return selected_features 

def Permutation_importance(X_test, y_test, classifier):

    # Determine the model based on the classifier name
    # Initialize the base model
    if classifier == "SVM":
        model = SVC(kernel='linear')
    elif classifier == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    elif classifier == "LogR":
        model = LogisticRegression(random_state=42)
    elif classifier == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42)
    elif classifier == "MLP":
        model = MLPClassifier(random_state=42)

    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    # Get the importances and sort them from most to least important
    importances = result.importances_mean
    indices = np.argsort(importances)[::-1]
    
    # Select features based on importance (threshold: features that have positive importance)
    selected_features = [i for i in indices if importances[i] > 0]  # Select features that have positive importance

    return selected_features

def RFE(X, y, n_features_to_select, classifier):
    # Determine the model based on the classifier name
    # Initialize the base model
    if classifier == "SVM":
        model = SVC(kernel='linear')
    elif classifier == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    elif classifier == "LogR":
        model = LogisticRegression(random_state=42)
    elif classifier == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42)
    elif classifier == "MLP":
        model = MLPClassifier(random_state=42)

    # Initialize RFE with the base model and the desired number of features to select
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)

    # Fit RFE
    rfe.fit(X, y)

    # Get the selected feature indices
    selected_features = np.where(rfe.support_)[0]

    return selected_features