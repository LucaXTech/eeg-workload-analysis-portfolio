"""
Spectral feature extraction example for EEG-based workload analysis.

This script provides a sanitized and self-contained example of the spectral
feature extraction approach used in the EEG workload project.

No participant data are included. If no input CSV is provided, the script
generates synthetic EEG-like data for demonstration purposes.

Expected CSV format, if using real data:
    Ch01, Ch02, Ch03, Ch04, Ch05, Ch06, Ch07, Ch08

Optional marker columns such as Ch09/Ch10 can be present, but are ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch


FS = 250  # Hz

EEG_CHANNELS = {
    "Ch01": "Fz",
    "Ch02": "C3",
    "Ch03": "Cz",
    "Ch04": "C4",
    "Ch05": "Pz",
    "Ch06": "PO7",
    "Ch07": "POz",
    "Ch08": "PO8",
}

FREQUENCY_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def compute_bandpower(
    signal: np.ndarray,
    fs: int,
    band: tuple[float, float],
    nperseg: int | None = None,
) -> float:
    """
    Compute absolute bandpower using Welch's method.

    Parameters
    ----------
    signal:
        One-dimensional EEG signal.
    fs:
        Sampling frequency in Hz.
    band:
        Frequency band as (low, high).
    nperseg:
        Segment length for Welch PSD estimation.

    Returns
    -------
    float
        Absolute power in the selected frequency band.
    """
    if nperseg is None:
        nperseg = min(len(signal), 2 * fs)

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    low, high = band
    idx = (freqs >= low) & (freqs <= high)

    if not np.any(idx):
        return np.nan

    return float(np.trapezoid(psd[idx], freqs[idx]))


def extract_window_features(
    window_df: pd.DataFrame,
    fs: int = FS,
) -> dict[str, float]:
    """
    Extract log-bandpower and ratio-based features from one EEG window.
    """
    features: dict[str, float] = {}

    for channel, electrode in EEG_CHANNELS.items():
        signal = window_df[channel].to_numpy(dtype=float)

        bandpowers = {}
        for band_name, band_range in FREQUENCY_BANDS.items():
            power = compute_bandpower(signal, fs=fs, band=band_range)
            bandpowers[band_name] = power

            features[f"{electrode}_{band_name}_logpower"] = np.log(power + 1e-12)

        theta = bandpowers["theta"]
        alpha = bandpowers["alpha"]
        beta = bandpowers["beta"]

        features[f"{electrode}_theta_alpha_ratio"] = theta / (alpha + 1e-12)
        features[f"{electrode}_alpha_theta_ratio"] = alpha / (theta + 1e-12)
        features[f"{electrode}_engagement_index"] = beta / (alpha + theta + 1e-12)

    return features


def extract_features_from_dataframe(
    eeg_df: pd.DataFrame,
    fs: int = FS,
    window_seconds: int = 16,
    step_seconds: int | None = None,
) -> pd.DataFrame:
    """
    Extract spectral features from fixed-length EEG windows.

    Parameters
    ----------
    eeg_df:
        DataFrame containing EEG channels Ch01-Ch08.
    fs:
        Sampling frequency in Hz.
    window_seconds:
        Length of each analysis window.
    step_seconds:
        Step between consecutive windows. If None, non-overlapping windows are used.

    Returns
    -------
    pd.DataFrame
        One row per window, with spectral features as columns.
    """
    missing_channels = [ch for ch in EEG_CHANNELS if ch not in eeg_df.columns]
    if missing_channels:
        raise ValueError(f"Missing EEG channels: {missing_channels}")

    if step_seconds is None:
        step_seconds = window_seconds

    window_samples = window_seconds * fs
    step_samples = step_seconds * fs

    rows = []

    for start in range(0, len(eeg_df) - window_samples + 1, step_samples):
        end = start + window_samples
        window_df = eeg_df.iloc[start:end]

        row = {
            "window_start_sample": start,
            "window_end_sample": end,
            "window_start_seconds": start / fs,
            "window_end_seconds": end / fs,
        }

        row.update(extract_window_features(window_df, fs=fs))
        rows.append(row)

    return pd.DataFrame(rows)


def generate_synthetic_eeg(
    duration_seconds: int = 64,
    fs: int = FS,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic EEG-like data for demonstration purposes only.
    """
    rng = np.random.default_rng(random_state)
    n_samples = duration_seconds * fs
    time = np.arange(n_samples) / fs

    data = {}

    for channel in EEG_CHANNELS:
        theta_component = 8e-6 * np.sin(2 * np.pi * 6 * time)
        alpha_component = 12e-6 * np.sin(2 * np.pi * 10 * time)
        beta_component = 5e-6 * np.sin(2 * np.pi * 20 * time)
        noise = 3e-6 * rng.standard_normal(n_samples)

        data[channel] = theta_component + alpha_component + beta_component + noise

    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract spectral EEG features from Ch01-Ch08 data."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Optional path to an input CSV file containing Ch01-Ch08.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="spectral_features_output.csv",
        help="Path where the extracted feature table will be saved.",
    )

    parser.add_argument(
        "--window_seconds",
        type=int,
        default=16,
        help="Length of each analysis window in seconds.",
    )

    args = parser.parse_args()

    if args.input_csv is None:
        print("No input CSV provided. Generating synthetic EEG-like data.")
        eeg_df = generate_synthetic_eeg()
    else:
        input_path = Path(args.input_csv)
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")
        eeg_df = pd.read_csv(input_path)

    features_df = extract_features_from_dataframe(
        eeg_df=eeg_df,
        fs=FS,
        window_seconds=args.window_seconds,
    )

    features_df.to_csv(args.output_csv, index=False)

    print(f"Extracted {len(features_df)} windows.")
    print(f"Saved features to: {args.output_csv}")
    print(features_df.head())


if __name__ == "__main__":
    main()
