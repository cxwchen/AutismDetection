import numpy as np
from scipy.signal import find_peaks
import pandas as pd

def pk_extract(x, time_values=None, height_threshold=0, prominence=1):
    """
    Extract and Compute statistics of peaks (if peaks exist).

    Returns:
            'num_peaks': int,
            'mean_amplitude': float,
            'max_amplitude': float,
            'mean_time_interval': float (if time_values provided)
    """
    peaks, properties = find_peaks(x, height=height_threshold, prominence=prominence)

    peaks_data = {
        'peak_amplitudes': x[peaks],
        'peak_indices': peaks
    }

    if time_values is not None:
        peaks_data['peak_times'] = time_values[peaks]

    stats = {
        'num_peaks': len(peaks_data['peak_amplitudes']),
        'mean_amplitude': np.mean(peaks_data['peak_amplitudes']) if peaks_data['peak_amplitudes'].size > 0 else 0,
        'max_amplitude': np.max(peaks_data['peak_amplitudes']) if peaks_data['peak_amplitudes'].size > 0 else 0
    }

    if 'peak_times' in peaks_data and len(peaks_data['peak_times']) > 1:
        time_intervals = np.diff(peaks_data['peak_times'])
        stats['mean_time_interval'] = np.mean(time_intervals)

    return stats, peaks_data


def pk_stats(list_of_timeseries, time_values=None):
    """
Takes list of timeseries and returns peak features per timeseries as a dataframe

Features: avg peak interval, max peak, avg peak amplitude
    """
    all_stats = []
    for i, ts in enumerate(list_of_timeseries):
        stats, peaks = pk_extract(ts, time_values)
        stats['series_id'] = i  # Track which series these stats belong to
        all_stats.append(stats)

    return pd.DataFrame(all_stats)