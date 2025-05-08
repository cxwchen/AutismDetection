#testing

from scipy.io import loadmat
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skfeature.function.similarity_based import reliefF as fs
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from skfeature import classifiers as cl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def main():
    # Load with encoding fix
    data = pd.read_csv('ABIDEII_Composite_Phenotypic.csv', encoding='ISO-8859-1')

    # Choose relevant features
    #columns_to_use = ['DX_GROUP', 'AGE_AT_SCAN ', 'SEX', 'FIQ', 'VIQ', 'PIQ', 'HANDEDNESS_CATEGORY'] #change to wanted features
    #data = data[columns_to_use]

    # Drop rows where DX_GROUP is missing
    data = data[data['DX_GROUP'].notna()]

    # Separate X and y
    X = data.drop(columns=['DX_GROUP'])
    y = data['DX_GROUP']
    # Encode categorical
    X = pd.get_dummies(X, drop_first=True)

    #drop the columns where all values are missing before the imputer
    X = X.dropna(axis=1, how='all')

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    #splitting the data in train and test 0.8:0.2 respecively
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 40)

    #performing lasso feature selection
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train, y_train)

    selected_features = np.where(lasso.coef_ != 0)[0]
    print(f"Selected features ({len(selected_features)}):", selected_features)

    #changing the test and train data to the data selected by the feature selection tool
    selected_train_x = X_train.iloc[:, selected_features]
    selected_test_X = X_test.iloc[:, selected_features]


    #scaling the features, otherwise the maximum iterations of the classifier will be exceded
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(selected_train_x)
    X_test_scaled = scaler.transform(selected_test_X)

    #applying the classifier from the classifier group
    y_pred = cl.applyLogR(X_train_scaled, X_test_scaled, y_train)

    #finding mse and accuracy
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy:', acc)

    #getting and printing the feature names
    feature_names = X_train.columns
    selected_feature_names = feature_names[selected_features]

    print(f"\nSelected feature names({len(selected_feature_names)}):")
    for name in selected_feature_names:
        print("-", name)

if __name__ == '__main__':
    main()
