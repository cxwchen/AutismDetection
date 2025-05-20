# -*- coding: utf-8 -*-
"""
Created on Fri May 16 12:58:56 2025

@author: kakis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wisardpkg import ClusWisard
from collections import defaultdict

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

addressSize = 40
minScore = 0.2
threshold = 10
discriminatorLimit = 100
n_runs = 80

accuracies = []
conf_matrices = []
report_sums = defaultdict(lambda: defaultdict(float))

for _ in range(n_runs):
    model = ClusWisard(addressSize, minScore, threshold, discriminatorLimit)
    model.train(X_train, y_train)
    predictions = model.classify(X_test)

    # Accuracy
    acc = accuracy_score(y_test, predictions)
    accuracies.append(acc)

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=np.unique(y_test))
    conf_matrices.append(cm)

    # Classification report
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                report_sums[label][metric] += value

# Averages
avg_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
avg_cm = np.mean(conf_matrices, axis=0)

avg_report = {
    label: {metric: value / n_runs for metric, value in metrics.items()}
    for label, metrics in report_sums.items()
}

# Display results
print(f"\nAverage Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")

# Format function again
def format_classification_report(avg_report, digits=2):
    headers = ["precision", "recall", "f1-score", "support"]
    rows = []
    for label in sorted(avg_report, key=lambda x: str(x)):
        row = [label]
        for h in headers:
            row.append(avg_report[label].get(h, 0.0))
        rows.append(row)

    # Create formatted string output
    output = "Classification report:\n"
    name_width = max(len(str(row[0])) for row in rows)
    width = digits + 6
    head_fmt = f"{{:>{name_width}s}} " + " ".join([f"{{:>{width}s}}" for _ in headers]) + "\n"
    row_fmt = f"{{:>{name_width}s}} " + " ".join([f"{{:>{width}.{digits}f}}" for _ in headers]) + "\n"
    output += head_fmt.format("", *headers)
    for row in rows:
        output += row_fmt.format(*row)
    return output

# Print nicely formatted report
formatted_report = format_classification_report(avg_report)
print(formatted_report)

print("\nAverage Confusion Matrix:")
print(avg_cm)
