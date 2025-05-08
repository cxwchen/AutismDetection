#testing

from scipy.io import loadmat
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from skfeature.function.similarity_based import reliefF as fs
from sklearn import svm
from sklearn.metrics import accuracy_score

mat = loadmat("COIL20.mat")

X = mat['X']
y = mat['Y'][:,0]

n_samples, n_features = np.shape(X)
n_labels = np.shape(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 40)

score = fs.reliefF(X_train, y_train)

idx = fs.feature_ranking(score)
acc_max = 0
feature = 0
for i in range(1, 50):
    num_fea = i
    selected_features_train = X_train[:, idx[0:num_fea]]
    selected_features_test = X_test[:, idx[0:num_fea]]

    clf = svm.LinearSVC()

    clf.fit(selected_features_train, y_train)
    y_predict = clf.predict(selected_features_test)

    acc = accuracy_score(y_test, y_predict)

    if acc >= acc_max:
        acc_max = acc
        feature = num_fea

print("Maximum accuracy:", acc_max)
print("Feature:", feature)
    