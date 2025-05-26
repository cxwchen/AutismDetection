from __future__ import division
import numpy as np
from sklearn.linear_model import Lasso, LassoLars
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SequentialFeatureSelector, VarianceThreshold
from scipy.stats import gamma

def rbf_dot(pattern1, pattern2, deg):
	size1 = pattern1.shape
	size2 = pattern2.shape

	G = np.sum(pattern1*pattern1, 1).reshape(size1[0],1)
	H = np.sum(pattern2*pattern2, 1).reshape(size2[0],1)

	Q = np.tile(G, (1, size2[0]))
	R = np.tile(H.T, (size1[0], 1))

	H = Q + R - 2* np.dot(pattern1, pattern2.T)

	H = np.exp(-H/2/(deg**2))

	return H


def hsic_gam(X, Y, alph = 0.5):
	"""
	X, Y are numpy vectors with row - sample, col - dim
	alph is the significance level
	auto choose median to be the kernel width
	"""
	n = X.shape[0]

	# ----- width of X -----
	Xmed = X

	G = np.sum(Xmed*Xmed, 1).reshape(n,1)
	Q = np.tile(G, (1, n) )
	R = np.tile(G.T, (n, 1) )

	dists = Q + R - 2* np.dot(Xmed, Xmed.T)
	dists = dists - np.tril(dists)
	dists = dists.reshape(n**2, 1)

	width_x = np.sqrt( 0.5 * np.median(dists[dists>0]) )
	# ----- -----

	# ----- width of X -----
	Ymed = Y

	G = np.sum(Ymed*Ymed, 1).reshape(n,1)
	Q = np.tile(G, (1, n) )
	R = np.tile(G.T, (n, 1) )

	dists = Q + R - 2* np.dot(Ymed, Ymed.T)
	dists = dists - np.tril(dists)
	dists = dists.reshape(n**2, 1)

	width_y = np.sqrt( 0.5 * np.median(dists[dists>0]) )
	# ----- -----

	bone = np.ones((n, 1), dtype = float)
	H = np.identity(n) - np.ones((n,n), dtype = float) / n

	K = rbf_dot(X, X, width_x)
	L = rbf_dot(Y, Y, width_y)

	Kc = np.dot(np.dot(H, K), H)
	Lc = np.dot(np.dot(H, L), H)

	testStat = np.sum(Kc.T * Lc) / n

	varHSIC = (Kc * Lc / 6)**2

	varHSIC = ( np.sum(varHSIC) - np.trace(varHSIC) ) / n / (n-1)

	varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

	K = K - np.diag(np.diag(K))
	L = L - np.diag(np.diag(L))

	muX = np.dot(np.dot(bone.T, K), bone) / n / (n-1)
	muY = np.dot(np.dot(bone.T, L), bone) / n / (n-1)

	mHSIC = (1 + muX * muY - muX - muY) / n

	al = mHSIC**2 / varHSIC
	bet = varHSIC*n / mHSIC

	thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]

	return (testStat, thresh)

def select_model(classifier):
    # Determine the model based on the classifier name
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
    else:
        raise ValueError("Unsupported classifier type")
    
    return model

def low_variance(X, threshold=0.01):
    print("Total features before low variance filter: ", X.shape[1])
    model = VarianceThreshold(threshold=threshold)
    X_reduced = model.fit_transform(X)
    selected_features = model.get_support(indices=True)
    print("Total features after low variance filter: ", X_reduced.shape[1])
    return selected_features

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

def LAND(X, y, lambda_reg=0.01):
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    n_samples, n_features = X.shape

    _, width_y = hsic_gam(y, y, alph=0.5)
    H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    L = rbf_dot(y, y, width_y)
    Lc = H @ L @ H
    L_vec = Lc.ravel()

    K_vecs = np.zeros((n_samples**2, n_features))
    for k in range(n_features):
        feature_k = X[:, k].reshape(-1, 1)
        _, width_x = hsic_gam(feature_k, feature_k, alph=0.5)
        K = rbf_dot(feature_k, feature_k, width_x)
        Kc = H @ K @ H
        K_vecs[:, k] = Kc.ravel()
    
    model = LassoLars(alpha=lambda_reg, fit_intercept=False)
    model.fit(K_vecs, L_vec)

    score = model.coef_
    selected_features = np.where(score != 0)[0]

    return selected_features

def lars_lasso(X, y, alpha=0.1, max_iter=100):
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

def Permutation_importance(X, y, classifier, select_features=None):

    # Determine the model based on the classifier name
    model = select_model(classifier)

    if select_features is not None:
        # Select the features based on the selected indices 
        X = X[:, select_features]

    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)

    # Get the importances and sort them from most to least important
    importances = result.importances_mean
    indices = np.argsort(importances)[::-1]
    
    # Select features based on importance (threshold: features that have positive importance)
    selected_features = [i for i in indices if importances[i] > 0]  # Select features that have positive importance

    return selected_features

def RecursiveFE(X, y, n_features_to_select, classifier, select_features=None):
    # Determine the model based on the classifier name
    model = select_model(classifier)

    if select_features is not None:
        # Select the features based on the selected indices 
        X = X[:, select_features]

    # Initialize RFE with the base model and the desired number of features to select
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)

    # Fit RFE
    rfe.fit(X, y)

    # Get the selected feature indices
    selected_features = np.where(rfe.support_)[0]

    return selected_features

def backwards_SFS(X, y, n_features_to_select, classifier, select_features=None):
    # Determine the model based on the classifier name
    model = select_model(classifier)

    if select_features is not None:
        # Select the features based on the selected indices 
        X = X[:, select_features]

    # Initialize SequentialFeatureSelector with the base model and the desired number of features to select
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select, direction='backward')

    # Fit SFS
    sfs.fit(X, y)

    # Get the selected feature indices
    selected_features = np.where(sfs.get_support())[0]

    return selected_features
