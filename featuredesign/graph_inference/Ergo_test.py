import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy.stats import levene,ks_2samp
from itertools import combinations
import statsmodels.api as sm


def ergodicity_test(fmri_data, n_windows=10):
    """
    Comprehensive ergodicity assessment for fMRI data with constant-series handling

    Returns:
    - Dictionary containing:
        'time_avg': Per-ROI time averages
        'ensemble_avg': Cross-ROI ensemble averages
        'ergodic_ratios': Time avg / Ensemble avg
        'stationarity_tests': ADF results
        'consistency_tests': Variance comparisons
        'constant_rois': List of constant ROIs
    """
    results = {
        'time_avg': [],
        'ensemble_avg': [],
        'ergodic_ratios': [],
        'stationarity_tests': [],
        'consistency_tests': {},
        'constant_rois': []
    }

    # Identify and exclude constant ROIs
    non_constant_mask = [not np.all(roi == roi[0]) for roi in fmri_data]
    fmri_filtered = [roi for roi, keep in zip(fmri_data, non_constant_mask) if keep]
    results['constant_rois'] = [f"ROI_{i}" for i, keep in enumerate(non_constant_mask) if not keep]

    if len(fmri_filtered) == 0:
        raise ValueError("All ROIs are constant - cannot perform ergodicity testing")

    # 1. Time vs Ensemble Averages
    time_means = [np.mean(roi) for roi in fmri_filtered]
    ensemble_means = np.mean(fmri_filtered, axis=1)  # Across ROIs at each timepoint

    # 2. Stationarity Tests (Prerequisite for ergodicity)
    adf_results = []
    for roi in fmri_filtered:
        try:
            adf_results.append(adfuller(roi))
        except ValueError:
            adf_results.append((np.nan, np.nan))  # Shouldn't happen due to filtering

    # 3. Variance Consistency (Bartlett's test & Levenes test)
    segments = np.array_split(fmri_filtered, n_windows, axis=1)
    bartlett_stat, bartlett_p = stats.bartlett(*[np.var(seg, axis=1) for seg in segments])
    levene_result = levene(*[np.var(seg, axis=1) for seg in segments])

    # 4. Ergodicity Critical Metrics
    results.update({
        'time_avg': time_means,
        'ensemble_avg': np.mean(ensemble_means),
        'ergodic_ratios': np.array(time_means) / np.mean(ensemble_means),
        'stationarity_tests': [x[1] for x in adf_results],  # p-values
        'consistency_tests': {
            'bartlett_p': bartlett_p,
            'levene_p': levene_result.pvalue,
            'variance_ratio': np.var(time_means) / np.var(ensemble_means)
        },
        'n_constant_rois': len(results['constant_rois'])
    })

    # 5. Distribution Consistency Tests (NEW)
    # Q-Q plot data collection (compare first 2 non-constant ROIs as example)
    qq_data = {
        'roi1': fmri_filtered[0] if len(fmri_filtered) > 0 else None,
        'roi2': fmri_filtered[1] if len(fmri_filtered) > 1 else None
    }

    # KS tests between all ROI pairs
    ks_results = []
    for (i, j) in combinations(range(len(fmri_filtered)), 2):
        stat, p = ks_2samp(fmri_filtered[i], fmri_filtered[j])
        ks_results.append({
            'roi_pair': (i, j),
            'ks_stat': stat,
            'ks_p': p
        })

    # Update results
    results.update({
        # [keep existing updates...],
        'distribution_tests': {
            'ks_results': ks_results,
            'percent_significant': 100 * np.mean([x['ks_p'] < 0.05 for x in ks_results]),
            'qq_sample_pair': qq_data
        }
    })
    return results


def plot_ergodicity(results):
    """Visualization for ergodicity assessment with constant ROI reporting"""
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    # Time vs Ensemble Averages
    ax1.scatter(range(len(results['time_avg'])), results['time_avg'],
                label='Per-ROI Time Averages')
    ax1.axhline(results['ensemble_avg'], color='r',
                label=f'Ensemble Average ({results["ensemble_avg"]:.2f})')
    ax1.set_title('Ergodicity Check: Time vs Ensemble Averages')
    ax1.set_ylabel('BOLD Signal')
    ax1.legend()

    # Ergodicity Ratios
    ax2.hist(results['ergodic_ratios'], bins=20)
    ax2.axvline(1, color='k', linestyle='--')
    ax2.set_title('Distribution of Time/Ensemble Ratios')
    ax2.set_xlabel('Ratio (Ideal = 1)')

    # Stationarity
    ax3.hist(results['stationarity_tests'], bins=20)
    ax3.axvline(0.05, color='r', linestyle='--')
    ax3.set_title('ADF Test p-values (Stationarity)')
    ax3.set_xlabel('p-value (p < 0.05 → stationary)')

    # Variance Consistency
    ax4.text(0.1, 0.5,
             f"Bartlett's p = {results['consistency_tests']['bartlett_p']:.3f}\n"
             f"Variance Ratio = {results['consistency_tests']['variance_ratio']:.3f}",
             fontsize=12)
    ax4.axis('off')
    ax4.set_title('Variance Homogeneity')

    # Constant ROI report
    if results['n_constant_rois'] > 0:
        ax5.text(0.1, 0.5,
                 f"Excluded {results['n_constant_rois']} constant ROIs:\n"
                 f"{', '.join(results['constant_rois'])}",
                 fontsize=12, color='red')
    else:
        ax5.text(0.1, 0.5, "No constant ROIs detected", fontsize=12)
    ax5.axis('off')
    ax5.set_title('Constant ROI Report')

    # 6. Distribution Tests Visualization (NEW)
    ax6 = fig.add_subplot(gs[2, :])  # Replaces old ax5

    # Q-Q Plot (if available)
    if results['distribution_tests']['qq_sample_pair']['roi1'] is not None:
        sm.qqplot_2samples(
            results['distribution_tests']['qq_sample_pair']['roi1'],
            results['distribution_tests']['qq_sample_pair']['roi2'],
            line='45',
            ax=ax6
        )
        ax6.set_title(f"Q-Q Plot: ROI1 vs ROI2 (Example Pair)")
    else:
        ax6.text(0.3, 0.5, "Insufficient ROIs for Q-Q comparison", fontsize=12)

    # KS Test Results
    ax6.text(1.1, 0.6,
             f"KS Test Results:\n"
             f"{results['distribution_tests']['percent_significant']:.1f}% pairs differ\n"
             f"(p < 0.05)",
             transform=ax6.transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))

    # [Keep constant ROI reporting if needed...]
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from AAL_test import load_files

    fmri_data = load_files()[0][0].T  # Assuming shape (ROIs, timepoints)

    try:
        results = ergodicity_test(fmri_data)
        plot_ergodicity(results)

        print("\nKey Ergodicity Metrics:")
        print(
            f"- % Stationary ROIs (ADF p<0.05): {100 * np.nanmean(np.array(results['stationarity_tests']) < 0.05):.1f}%")
        print(
            f"- Mean Ergodicity Ratio: {np.nanmean(results['ergodic_ratios']):.3f} ± {np.nanstd(results['ergodic_ratios']):.3f}")
        print(f"- Variance Homogeneity p-value: {results['consistency_tests']['bartlett_p']:.4f}")
        print(f"- Levene p-value: {results['consistency_tests']['levene_p']:.4f}")
        if results['n_constant_rois'] > 0:
            print(f"\nWarning: Excluded {results['n_constant_rois']} constant ROIs from analysis")
        print("\nKey Ergodicity Metrics:")
        print(f"- % ROI pairs with different distributions (KS test): "
              f"{results['distribution_tests']['percent_significant']:.1f}%")

    except ValueError as e:
        print(f"Analysis failed: {str(e)}")