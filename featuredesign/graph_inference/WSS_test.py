import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

def analyze_fmri_roi(fmri_series, window_size=50, step_size=10, TR=0.2, plot=False, n_lags=1):
    """
    Analyze stationarity and autocorrelation properties of an fMRI ROI time series.

    Parameters:
    - fmri_series: 1D numpy array of fMRI time series for a single ROI
    - window_size: Size of rolling window for variance/mean calculations
    - step_size: Step between consecutive windows
    - TR: Repetition time (for autocorrelation calculations)

    Returns:
    - Dictionary containing:
        'rolling_stats': DataFrame of windowed statistics
        'acf_df': DataFrame of autocorrelation results
        'stationarity_tests': Dictionary of test results
        'plots': Dictionary of matplotlib figure objects
    """

    # Initialize output containers
    results = {
        'rolling_stats': None,
        'acf_df': None,
        'stationarity_tests': {},
        'plots': {}
    }

    # 1. Rolling window statistics
    variances = []
    means = []
    for i in range(0, len(fmri_series) - window_size + 1, step_size):
        window = fmri_series[i:i + window_size]
        variances.append(np.var(window))
        means.append(np.mean(window))

    index_labels = [f"Sample {i} to {i + window_size}"
                    for i in range(0, len(fmri_series) - window_size + 1, step_size)]
    results['rolling_stats'] = pd.DataFrame({
        'Variance': variances,
        'Mean': means
    }, index=index_labels)

    # 2. Stationarity tests
    if not np.all(fmri_series == fmri_series[0]):  # Skip if constant
        results['stationarity_tests']['ADF'] = ts.adfuller(fmri_series)[1]  # ADF p-value
        results['stationarity_tests']['Ljung-Box'] = acorr_ljungbox(
            fmri_series, lags=10 * TR, return_df=True)
    else:
        results['stationarity_tests']['ADF'] = np.nan
        results['stationarity_tests']['Ljung-Box'] = pd.DataFrame()

    # 3. Autocorrelation analysis
    rho = ts.acf(fmri_series, nlags=10 * TR)[n_lags] if not np.all(fmri_series == fmri_series[0]) else np.nan
    results['acf_df'] = pd.DataFrame({
        'ACF_lag1': [rho],
        'ROI': ['Current_ROI']
    })
    if plot:
        # 4. Create plots
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot Variance
        results['rolling_stats']['Variance'].plot(ax=ax1, color='blue', label='Variance')
        mean_var = np.mean(variances)
        ax1.axhline(y=mean_var, color='red', linestyle='--',
                    label=f'Mean Variance = {mean_var:.2f}')
        ax1.set_title('Rolling Window Statistics')
        ax1.set_ylabel('Variance')
        ax1.legend()
        ax1.grid(True)

        # Plot Mean
        results['rolling_stats']['Mean'].plot(ax=ax2, color='green', label='Mean')
        mean_val = np.mean(means)
        ax2.axhline(y=mean_val, color='orange', linestyle='--',
                    label=f'Mean Value = {mean_val:.2f}')
        ax2.set_ylabel('Mean Signal')
        ax2.set_xlabel('Window')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        results['plots']['rolling_stats'] = fig1

        # ACF Plot
        fig2, ax3 = plt.subplots(figsize=(10, 4))
        plot_acf(fmri_series, lags=10 * TR, ax=ax3)
        ax3.set_title('Autocorrelation Function')
        results['plots']['acf_plot'] = fig2

    return results


def WSS_test(fmri_data, window_size=50, step_size=10, TR=2, n_lags=1):
    all_results = []
    all_variances = []
    all_means = []
    all_acfs = []

    for i, roi_series in enumerate(fmri_data):
        roi_result = analyze_fmri_roi(roi_series, window_size, step_size, TR, n_lags=n_lags)
        roi_result['acf_df']['ROI'] = f'ROI_{i + 1}'
        all_results.append(roi_result)

        # Collect data for averages
        if not np.all(roi_series == roi_series[0]):
            all_variances.append(roi_result['rolling_stats']['Variance'])
            all_means.append(roi_result['rolling_stats']['Mean'])
            all_acfs.append(roi_result['acf_df']['ACF_lag1'].values[0])

    # Plot averages
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Plot average variance across ROIs
    avg_variance = pd.concat(all_variances, axis=1).mean(axis=1)
    avg_variance.plot(ax=ax1, color='blue', label='Average Variance')
    ax1.axhline(y=avg_variance.mean(), color='red', linestyle='--',
                label=f'Overall Mean = {avg_variance.mean():.2f}')
    ax1.set_title('Average Variance Across ROIs')
    ax1.set_ylabel('Variance')
    ax1.legend()
    ax1.grid(True)

    # Plot average mean across ROIs
    avg_mean = pd.concat(all_means, axis=1).mean(axis=1)
    avg_mean.plot(ax=ax2, color='green', label='Average Mean')
    ax2.axhline(y=avg_mean.mean(), color='orange', linestyle='--',
                label=f'Overall Mean = {avg_mean.mean():.2f}')
    ax2.set_ylabel('Mean Signal')
    ax2.legend()
    ax2.grid(True)

    # Plot ACF distribution
    ax3.hist(all_acfs, bins=20, color='purple', alpha=0.7)
    ax3.axvline(x=np.mean(all_acfs), color='black', linestyle='--',
                label=f'Mean ACF = {np.mean(all_acfs):.2f}')
    ax3.set_title(f'Distribution of Lag-{n_lags} Autocorrelations')
    ax3.set_xlabel('ACF Value')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    return all_results


# Modified main block
if __name__ == "__main__":
    from AAL_test import load_files

    n_lags=5 # number of lags for autocorrelation
    fmri_data = load_files()[0][0].T # fmri_data per subject
    all_results = WSS_test(fmri_data,n_lags=n_lags)

    # Example: Print summary statistics
    print("\n=== Summary Statistics Across ROIs ===")
    print(f"Average ADF p-value: {np.nanmean([res['stationarity_tests']['ADF'] for res in all_results]):.4f}")
    print(f"Average Lag-{n_lags} Autocorrelation: {np.nanmean([res['acf_df']['ACF_lag1'].values[0] for res in all_results]):.4f}")


    # Access individual ROI results (unchanged)
    roi_num = 0  # Change this to view different ROIs
    print(f"\n=== Detailed results for ROI {roi_num + 1} ===")
    print("Rolling Statistics:\n", all_results[roi_num]['rolling_stats'])
    print("\nAutocorrelation:\n", all_results[roi_num]['acf_df'])

    print("\nADF Stationarity Test Results (α = 0.05):")
    for i, res in enumerate(all_results):
        adf_pval = res['stationarity_tests']['ADF']
        if np.isnan(adf_pval):
            print(f"ROI {i + 1}: ADF p-value = NaN (skipped or constant series)")
        elif adf_pval < 0.05:
            print(f"ROI {i + 1}: ADF p-value = {adf_pval:.4f} → REJECT null hypothesis (stationary ✅)")
        else:
            print(f"ROI {i + 1}: ADF p-value = {adf_pval:.4f} → FAIL TO REJECT null hypothesis (non-stationary ⚠️)")

    for i, roi_series in enumerate(fmri_data[:10]):  # Limit to first 10
        plt.figure(figsize=(6, 3))
        plot_acf(roi_series, lags=20)
        plt.title(f"ROI {i + 1} ACF")
        plt.show()