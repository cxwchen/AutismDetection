import scipy.io
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from skfeature.function.wrapper import svm_backward as fs
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np


def main():
    # load data
    mat = scipy.io.loadmat('COIL20.mat')
    X = mat['X'].astype(float)  # data
    y = mat['Y'][:, 0]          # labels as 1D array
    n_samples, n_features = X.shape

    # split data into 5 folds
    #ss = KFold(n_splits=5, shuffle=True, random_state=40)

    # Use train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    clf = svm.LinearSVC(max_iter=10000)  # linear SVM with higher max_iter

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
    idx = fs.svm_backward(X_train, y_train, max_k)

    for k in range(1, max_k + 1):
        top_k = idx[:k]
        clf.fit(X_train[:, top_k], y_train)
        y_pred = clf.predict(X_test[:, top_k])
        acc = accuracy_score(y_test, y_pred)
        acc_matrix[k - 1].append(acc)

    
    avg_acc = [np.mean(accs) for accs in acc_matrix]
    best_k = np.argmax(avg_acc) +1
    best_acc = avg_acc[best_k - 1]

    # Output results
    print("\nAverage Accuracies by k:")
    for k, acc in enumerate(avg_acc, 1):
        print(f"k={k}: {acc:.4f}")

    print(f"\n Best number of features: {best_k} with accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()