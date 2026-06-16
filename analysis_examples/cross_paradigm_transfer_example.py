"""
Cross-paradigm transfer example for EEG-based workload decoding.

This script provides a sanitized and self-contained example of how a workload
classifier can be trained on one experimental paradigm and tested on another.

Main goals:
- simulate two related EEG workload paradigms;
- train on one paradigm and test on the other;
- compare within-paradigm and cross-paradigm performance;
- use leakage-aware scikit-learn pipelines;
- evaluate Logistic Regression and shrinkage LDA.

No participant data are included. If no input CSV is provided, the script
generates synthetic feature data for demonstration purposes.

Expected input CSV format, if using real features:
    - one row per EEG window or segment;
    - numeric EEG feature columns;
    - paradigm column, for example: N_LEVELS or N_SPEED;
    - label column, for example: low or high;
    - optional subject_id column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

ELECTRODES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"]
BANDS = ["delta", "theta", "alpha", "beta"]

NON_FEATURE_COLUMNS = {
    "subject_id",
    "paradigm",
    "condition",
    "label",
    "session",
    "block",
    "window_id",
    "segment_idx",
    "window_start_sample",
    "window_end_sample",
    "window_start_seconds",
    "window_end_seconds",
}


def generate_synthetic_cross_paradigm_data(
    n_subjects: int = 15,
    windows_per_condition: int = 12,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate synthetic EEG features for two related paradigms.

    The synthetic dataset includes:
    - N_LEVELS: low vs high workload simulated as 1-back vs 3-back;
    - N_SPEED: low vs high workload simulated as slow vs fast 2-back.

    Both paradigms share a workload-related structure, but each paradigm also
    has a small domain-specific shift. This makes cross-paradigm transfer
    possible but harder than within-paradigm classification.
    """
    rng = np.random.default_rng(random_state)

    rows = []

    paradigms = {
        "N_LEVELS": {
            "low_condition": "1-back",
            "high_condition": "3-back",
            "domain_shift": 0.00,
        },
        "N_SPEED": {
            "low_condition": "slow",
            "high_condition": "fast",
            "domain_shift": 0.20,
        },
    }

    for subject_idx in range(1, n_subjects + 1):
        subject_id = f"S{subject_idx:02d}"
        subject_offset = rng.normal(loc=0.0, scale=0.30)

        for paradigm, info in paradigms.items():
            for label, condition in [
                ("low", info["low_condition"]),
                ("high", info["high_condition"]),
            ]:
                for window_idx in range(windows_per_condition):
                    row = {
                        "subject_id": subject_id,
                        "paradigm": paradigm,
                        "condition": condition,
                        "label": label,
                        "window_id": window_idx,
                    }

                    for electrode in ELECTRODES:
                        electrode_shift = rng.normal(loc=0.0, scale=0.05)

                        for band in BANDS:
                            value = rng.normal(loc=0.0, scale=0.35)
                            value += subject_offset
                            value += electrode_shift
                            value += info["domain_shift"]

                            if label == "high":
                                if band == "theta":
                                    value += 0.45
                                elif band == "alpha":
                                    value -= 0.35
                                elif band == "beta":
                                    value += 0.25

                            if paradigm == "N_SPEED":
                                if band == "beta":
                                    value += 0.15
                                elif band == "alpha":
                                    value -= 0.10

                            row[f"{electrode}_{band}_logpower"] = value

                        theta_power = np.exp(row[f"{electrode}_theta_logpower"])
                        alpha_power = np.exp(row[f"{electrode}_alpha_logpower"])
                        beta_power = np.exp(row[f"{electrode}_beta_logpower"])

                        row[f"{electrode}_theta_alpha_ratio"] = theta_power / (
                            alpha_power + 1e-12
                        )
                        row[f"{electrode}_alpha_theta_ratio"] = alpha_power / (
                            theta_power + 1e-12
                        )
                        row[f"{electrode}_engagement_index"] = beta_power / (
                            alpha_power + theta_power + 1e-12
                        )

                    rows.append(row)

    return pd.DataFrame(rows)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select numeric feature columns while excluding identifiers and labels.
    """
    feature_columns = []

    for column in df.columns:
        if column in NON_FEATURE_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)

    if not feature_columns:
        raise ValueError("No numeric feature columns found.")

    return feature_columns


def build_models() -> dict[str, Pipeline]:
    """
    Build leakage-aware classification pipelines.

    Scaling is fitted only on the training data inside each pipeline.
    """
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "shrinkage_lda": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearDiscriminantAnalysis(
                        solver="lsqr",
                        shrinkage="auto",
                    ),
                ),
            ]
        ),
    }


def evaluate_transfer(
    df: pd.DataFrame,
    feature_columns: list[str],
    train_paradigm: str,
    test_paradigm: str,
    label_col: str = "label",
    paradigm_col: str = "paradigm",
) -> list[dict[str, object]]:
    """
    Train models on one paradigm and test them on another.
    """
    train_df = df[df[paradigm_col] == train_paradigm].copy()
    test_df = df[df[paradigm_col] == test_paradigm].copy()

    if train_df.empty:
        raise ValueError(f"No rows found for train paradigm: {train_paradigm}")

    if test_df.empty:
        raise ValueError(f"No rows found for test paradigm: {test_paradigm}")

    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df[label_col].to_numpy()

    X_test = test_df[feature_columns].to_numpy(dtype=float)
    y_test = test_df[label_col].to_numpy()

    results = []

    models = build_models()

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append(
            {
                "model": model_name,
                "train_paradigm": train_paradigm,
                "test_paradigm": test_paradigm,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro"),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(
                    y_test,
                    y_pred,
                    output_dict=True,
                ),
            }
        )

    return results


def print_results(results: list[dict[str, object]]) -> None:
    """
    Print compact transfer results.
    """
    for result in results:
        print("\n" + "=" * 80)
        print(
            f"{result['model']} | "
            f"train: {result['train_paradigm']} -> "
            f"test: {result['test_paradigm']}"
        )
        print("=" * 80)
        print(f"Training samples:    {result['n_train']}")
        print(f"Test samples:        {result['n_test']}")
        print(f"Accuracy:            {result['accuracy']:.3f}")
        print(f"Balanced accuracy:   {result['balanced_accuracy']:.3f}")
        print(f"Macro F1-score:      {result['macro_f1']:.3f}")
        print("Confusion matrix:")
        print(np.asarray(result["confusion_matrix"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-paradigm transfer example for EEG workload decoding."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Optional path to a CSV feature table.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="cross_paradigm_results.csv",
        help="Path where compact transfer results will be saved.",
    )

    parser.add_argument(
        "--paradigm_col",
        type=str,
        default="paradigm",
        help="Name of the paradigm column.",
    )

    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Name of the workload label column.",
    )

    args = parser.parse_args()

    if args.input_csv is None:
        print("No input CSV provided. Generating synthetic cross-paradigm data.")
        df = generate_synthetic_cross_paradigm_data()
    else:
        input_path = Path(args.input_csv)

        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")

        df = pd.read_csv(input_path)

    required_columns = {args.paradigm_col, args.label_col}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    available_paradigms = sorted(df[args.paradigm_col].unique())

    if len(available_paradigms) < 2:
        raise ValueError(
            "At least two paradigms are required for cross-paradigm transfer."
        )

    feature_columns = get_feature_columns(df)

    print(f"Loaded table shape: {df.shape}")
    print(f"Available paradigms: {available_paradigms}")
    print(f"Number of features: {len(feature_columns)}")

    all_results = []

    for train_paradigm in available_paradigms:
        for test_paradigm in available_paradigms:
            results = evaluate_transfer(
                df=df,
                feature_columns=feature_columns,
                train_paradigm=train_paradigm,
                test_paradigm=test_paradigm,
                label_col=args.label_col,
                paradigm_col=args.paradigm_col,
            )

            all_results.extend(results)

    print_results(all_results)

    compact_results = pd.DataFrame(
        [
            {
                "model": result["model"],
                "train_paradigm": result["train_paradigm"],
                "test_paradigm": result["test_paradigm"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "accuracy": result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "macro_f1": result["macro_f1"],
            }
            for result in all_results
        ]
    )

    compact_results.to_csv(args.output_csv, index=False)

    print("\nCompact results:")
    print(compact_results)

    print(f"\nSaved compact results to: {args.output_csv}")


if __name__ == "__main__":
    main()
