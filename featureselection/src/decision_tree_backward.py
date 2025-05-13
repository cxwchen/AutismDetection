import scipy.io
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from skfeature.function.wrapper import decision_tree_backward as fs
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from skfeature import classifiers as clas
from sklearn.impute import SimpleImputer

def main():
    # Load with encoding fix
    data = pd.read_csv('ABIDEII_Composite_Phenotypic.csv', encoding='ISO-8859-1')

    # Choose relevant features
    columns_to_use = ['DX_GROUP', 'AGE_AT_SCAN ', 'SEX', 'FIQ', 'VIQ', 'PIQ', 'HANDEDNESS_CATEGORY'] #change to wanted features
    data = data[columns_to_use]

    # Drop rows where DX_GROUP is missing
    data = data[data['DX_GROUP'].notna()]

    # Separate X and y
    X = data.drop(columns=['DX_GROUP'])
    y = data['DX_GROUP']

    # Encode categorical
    X = pd.get_dummies(X, drop_first=True)

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


    #X = mat['X'].astype(float)  # data
    #y = mat['Y'][:, 0]          # labels as 1D array
    #n_samples, n_features = X.shape

    # split data into 5 folds
    #ss = KFold(n_splits=5, shuffle=True, random_state=40)

    # Use train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    max_k = 10
    acc_matrix = [[] for _ in range(max_k)]


#    for train_idx, test_idx in ss.split(X, y):
        
#        # select features from training data
 #       idx = fs.svm_backward(X[train_idx], y[train_idx], max_k)
#
#        for k in range(1, max_k + 1):
    #        top_k = idx[:k]
   #         clf.fit(X[train_idx][:, top_k], y[train_idx])
  #          y_pred = clf.predict(X[test_idx][:, top_k])
 #           acc = accuracy_score(y[test_idx], y_pred)
#            acc_matrix[k - 1].append(acc)

        
    # select features from training data
    idx = fs.decision_tree_backward(X_train, y_train, max_k)

    best_features = []

    for k in range(1, max_k + 1):
        top_k = idx[:k]

        # Get the feature names corresponding to the selected features
        top_k_feature_names = X_train.columns[top_k].tolist()
        best_features.append((k, top_k_feature_names))

        X_train_top_k = X_train.iloc[:, top_k]
        X_test_top_k = X_test.iloc[:, top_k]

        model = clas.applySVM(X_train_top_k, y_train)
        y_predict = model.predict(X_test_top_k)
        acc = accuracy_score(y_test, y_predict)
        acc_matrix[k - 1].append(acc)

    
    avg_acc = [np.mean(accs) for accs in acc_matrix]
    best_k = np.argmax(avg_acc) +1
    best_acc = avg_acc[best_k - 1]

    # Output results
    print("\nAverage Accuracies by k:")
    for k, acc in enumerate(avg_acc, 1):
        print(f"k={k}: {acc:.4f}")

    print(f"\n Best number of features: {best_k} with accuracy: {best_acc:.4f}")
    print(f"\n Best features for k = {best_k}: {best_features[best_k - 1][1]}")

if __name__ == '__main__':
    main()