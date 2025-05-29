####### This file is for single site classification on NYU #########
import os
import sys
import datetime
import pandas as pd
from stratiKFold import runCV

# Create a timestamped log file
os.makedirs('logs', exist_ok=True)
log_filename = f'logs/run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_file = open(log_filename, 'w')

# Redirect all prints to the log file and still see them in the terminal
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, log_file)

female_df = pd.read_csv("ourfeats_female.csv.gz")
male_df = pd.read_csv("ourfeats_male.csv.gz")

def run_singlesite(sitename="NYU"):
    comb_df = pd.concat([female_df, male_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    comb_df = comb_df[comb_df['SITE_ID'] == sitename].reset_index(drop=True)
    runCV(comb_df, label="combined_onlyNYU")

if __name__ == "__main__":
    run_singlesite()