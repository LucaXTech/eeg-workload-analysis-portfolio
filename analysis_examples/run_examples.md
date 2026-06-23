# Running the Analysis Examples

This document provides example commands for running the sanitized analysis scripts included in this folder.

The scripts can run without participant data by generating synthetic EEG-like signals or synthetic feature tables when no input file is provided.

The synthetic outputs are included only to demonstrate the workflow structure. They are not intended to reproduce the original MSc thesis results.

---

## Setup

The analysis examples only require the packages listed in `requirements.txt`.

From the repository root, install the analysis dependencies with:

```bash
pip install -r requirements.txt
```

The acquisition scripts are separate from the analysis examples. They require the additional dependencies listed in `requirements_acquisition.txt`:

```bash
pip install -r requirements_acquisition.txt
```

You only need `requirements_acquisition.txt` if you want to inspect or run the g.Pype-based acquisition scripts.

---

## Spectral feature extraction

Run the spectral feature extraction example with synthetic EEG-like data:

```bash
python analysis_examples/spectral_features_example.py
```

This command generates synthetic EEG-like data, extracts spectral features and saves the output table as:

```text
spectral_features_output.csv
```

To use a custom EEG CSV file:

```bash
python analysis_examples/spectral_features_example.py --input_csv path/to/eeg_file.csv --output_csv spectral_features_output.csv
```

Expected EEG columns:

```text
Ch01, Ch02, Ch03, Ch04, Ch05, Ch06, Ch07, Ch08
```

Additional marker columns such as `Ch09` or `Ch10` may be present, but they are ignored by the example script.

---

## Baseline classification

Run the baseline classification example with synthetic feature data:

```bash
python analysis_examples/baseline_classification_example.py
```

This command generates a synthetic EEG feature table and evaluates:

* Logistic Regression;
* shrinkage Linear Discriminant Analysis;
* stratified holdout validation;
* leave-one-subject-out validation.

To use a custom feature table:

```bash
python analysis_examples/baseline_classification_example.py --input_csv path/to/features.csv
```

Expected columns:

```text
subject_id, label, numeric feature columns
```

The default label column is:

```text
label
```

The default subject/group column is:

```text
subject_id
```

---

## Time-on-task analysis

Run the time-on-task example with synthetic temporal feature data:

```bash
python analysis_examples/time_on_task_example.py
```

This command generates synthetic feature data, performs temporal trend analysis and saves output files in:

```text
time_on_task_outputs/
```

Generated outputs include:

```text
segment_summary.csv
spearman_temporal_trends.csv
early_middle_late_tests.csv
*_timecourse.png
```

To use a custom feature table:

```bash
python analysis_examples/time_on_task_example.py --input_csv path/to/features.csv
```

Expected columns:

```text
subject_id, condition, segment_idx, numeric feature columns
```

If `segment_idx` is not available, the script can infer it from:

```text
window_start_seconds
```

or:

```text
window_start_sample
```

---

## Cross-paradigm transfer

Run the cross-paradigm transfer example with synthetic feature data:

```bash
python analysis_examples/cross_paradigm_transfer_example.py
```

This command generates synthetic data for two related paradigms and evaluates both within-paradigm and cross-paradigm classification.

The compact result table is saved as:

```text
cross_paradigm_results.csv
```

To use a custom feature table:

```bash
python analysis_examples/cross_paradigm_transfer_example.py --input_csv path/to/features.csv
```

Expected columns:

```text
paradigm, label, numeric feature columns
```

Optional columns such as `subject_id`, `condition`, `window_id` or `segment_idx` may also be included.

---

## Notes

The scripts are designed as sanitized analysis examples and can be executed without access to the original private dataset.

Synthetic data are generated only to make the examples executable and to demonstrate the expected workflow structure.

The outputs produced from synthetic data are not thesis results and should not be interpreted as participant-derived findings.

No raw EEG recordings, participant identifiers, questionnaire data or behavioral response files are included in this repository.
