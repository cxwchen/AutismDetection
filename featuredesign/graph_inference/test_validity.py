from nilearn.interfaces.fmriprep import load_confounds
from AAL_test import multiset_feats, load_files
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.regression.linear_model import GLSAR
from nilearn.signal import clean
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import statsmodels.tsa.stattools as ts
import pandas as pd
import matplotlib.pyplot as plt

# Initialize
fmri_data = load_files()[0][0].T
TR = 2 # s
x_bold = fmri_data[0]

# Test WSS
variances = []
means = []
n_window = 50
n_step = 10

#p_value = ts.adfuller(fmri_data[0])  # p < 0.05 → stationair

for i in range(0, len(fmri_data[0]) - n_window + 1, n_step):
    variance = np.var(fmri_data[0][i:i+n_window], axis=0)
    mean = np.mean(fmri_data[0][i:i+n_window], axis=0)
    variances.append(variance)
    means.append(mean)

index_labels = [f"Sample {i} to {i+n_window}" for i in range(0, len(fmri_data[0]) - n_window + 1, n_step)]
df_var = pd.DataFrame(list(zip(variances, means)), index=index_labels, columns=["Variance","Mean"])

# Test Mixing
lb_test = acorr_ljungbox(fmri_data[0], lags=10*TR, return_df=True)
plot_acf(fmri_data[0], lags=10*TR)

# Test autocorrelation
rhos = []

for row in fmri_data:
    rho = ts.acf(row, nlags=1)[1]
    rhos.append(rho)
df_rho = pd.DataFrame(rhos, columns=["ACF"])
df_rho.insert(0, 'ROI',  [f"ROI_{i + 1}" for i in range(len(df_rho))])
df_rho.reset_index()

####################################
#   MAIN    #
####################################

print("data:\n", fmri_data)
print(lb_test)
print("Variances per block:\n", df_var)
print("Autocorrelation on Lag 1:\n", df_rho)

# Plot directly from DataFrame
df_rho.plot()
mean_rho = df_rho["ACF"].mean()
plt.axhline(y=mean_rho, color='black', linestyle='--', label=f"Mean = {mean_rho:.2f}")
plt.xlabel('Time')
plt.ylabel('C(x)')
plt.grid(True)

df_var.plot()
mean_val = df_var["Variance"].mean()
plt.axhline(y=mean_val, color='black', linestyle='--', label=f"Mean = {mean_val:.2f}")
plt.xlabel("Sample Window")
plt.ylabel("Variance")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()