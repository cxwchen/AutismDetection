# -*- coding: utf-8 -*-
"""
Created on Fri May 16 12:12:35 2025

@author: kakis
"""
from itertools import product
from wisardpkg import ClusWisard
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load and shuffle data
df = pd.read_csv("female_df_merged.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop subject_id, extract labels
df.drop(columns=['subject_id'], inplace=True)
y = df['DX_GROUP'].astype(str).tolist()
X = df.drop(columns=['DX_GROUP'])

# Check for missing values
missing = X.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    print("Missing values detected:")
    print(missing)
else:
    print("No missing values detected.")

# Thermometer encoding
def thermometer_encoding(data, n_bits=6):
    data = np.asarray(data)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    norm = (data - min_vals) / (max_vals - min_vals + 1e-9)
    binary_data = []
    for sample in norm:
        encoded_sample = []
        for value in sample:
            bits = [1 if i < value * n_bits else 0 for i in range(n_bits)]
            encoded_sample.extend(bits)
        binary_data.append(encoded_sample)
    return np.array(binary_data)

# Encode features
X_encoded = thermometer_encoding(X.values, n_bits=6)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded.tolist(), y, test_size=0.2, random_state=42
)
print("...")
# Define your parameter grid
param_grid = {
    "minScore": np.arange(4,8,1)/10,
    "threshold": np.arange(7,14,1),
    "discriminatorLimit": np.arange(90,200,20)
}
print("...")
# Generate all combinations of parameters
param_combinations = list(product(
    param_grid["minScore"],
    param_grid["threshold"],
    param_grid["discriminatorLimit"]
))

print("...")

best_score = 0
best_params = None
results = []

# Number of runs per combination to smooth randomness
n_runs = 20
i =0
k=0
for score, thresh, limit in param_combinations:
    accuracies = []
    i+=1
    if (i%100==0):
        print(f"40, {score}, {thresh}, {limit}")
    for _ in range(n_runs):
        model = ClusWisard(40, score, thresh, limit)
        model.train(X_train, y_train)
        preds = model.classify(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
    avg_acc = np.mean(accuracies)
    results.append((avg_acc, (40, score, thresh, limit)))
    #print(f"Tested: addressSize={addr}, minScore={score}, threshold={thresh}, limit={limit} → Avg Accuracy: {avg_acc:.4f}")
    if avg_acc > best_score:
        k+=1
        best_score = avg_acc
        best_params = (40, score, thresh, limit)
        if (k%10==0):
            print(f"40, {score}, {thresh}, {limit}: {avg_acc}")

print("\nBest Parameters:")
print(f"addressSize = {best_params[0]}")
print(f"minScore = {best_params[1]}")
print(f"threshold = {best_params[2]}")
print(f"discriminatorLimit = {best_params[3]}")
print(f"Average Accuracy = {best_score:.4f}")

# print("Clusters per class:")
# for label in model.getTrainedClasses():
#     print(f"{label}: {len(model.getDiscriminators(label))} clusters")
# 
    
# model = ClusWisard(..., returnActivationDegree=True)
# outputs = model.classify(X_test)
# print(outputs)  # Each value shows the number of active RAMs per prediction


