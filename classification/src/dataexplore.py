import pandas as pd
import numpy as np

female_df = pd.read_csv("female_df_merged.csv")

# Count values in the DX_group column
counts = female_df['DX_GROUP'].value_counts()

# Print the results
print(f"Number of subjects with autism: {counts.get(1, 0)}")
print(f"Number of control subjects: {counts.get(0, 0)}")

print("Missing values per column:")

ourfeats = pd.read_csv('ourfeats.csv.gz', compression='gzip')
print(ourfeats.isnull().sum())

# 2. Count totals for ASD and Control (label 1 = ASD, 0 = control)
total_asd = (ourfeats['label'] == 1).sum()
total_control = (ourfeats['label'] == 0).sum()
print(f"\nTotal ASD subjects: {total_asd}")
print(f"Total Control subjects: {total_control}")

# 3. Count by sex and label
counts = ourfeats.groupby(['sex', 'label']).size().unstack(fill_value=0)
counts.index = counts.index.map({'M':'Male', 'F':'Female'})  # nicer labels if you want

print("\nCounts by sex and diagnosis:")
print(counts)

print(ourfeats.columns.tolist())