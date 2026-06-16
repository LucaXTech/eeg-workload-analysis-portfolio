# Reproducibility Notes

This document summarizes the main reproducibility and validation principles followed in the EEG workload analysis workflow.

The repository is a sanitized portfolio version of the original MSc thesis workflow. It is designed to document the computational structure of the project without sharing participant data or private analysis outputs.

---

## Data privacy

No participant-level data are included in this repository.

The repository does not contain:

* raw EEG recordings;
* behavioral response files;
* questionnaire data;
* participant identifiers;
* subject-level output folders;
* private thesis output directories;
* clinical or sensitive data.

The analysis scripts are provided as representative examples and can run with synthetic data where appropriate.

---

## Data leakage prevention

EEG classification pipelines are particularly sensitive to data leakage because multiple windows can come from the same subject, session or task block.

To reduce overly optimistic results, the analysis workflow follows these principles:

* preprocessing parameters should be estimated only from training data when they are part of the model pipeline;
* scaling is performed inside scikit-learn pipelines;
* train/test separation should be defined before model fitting;
* subject-level splits should be used when evaluating cross-subject generalization;
* windows from the same subject should not be treated as fully independent samples in cross-subject validation;
* feature selection or normalization should not use information from the test set.

The example classification scripts use scikit-learn `Pipeline` objects so that scaling is fitted only on the training split.

---

## Validation strategies

The original analysis workflow included several complementary validation strategies.

### Within-subject validation

Within-subject validation evaluates whether models can generalize across separate sessions or blocks from the same participant.

This is useful for assessing intra-individual decoding performance, but it does not prove that the model generalizes to unseen participants.

### Leave-one-subject-out validation

Leave-one-subject-out validation evaluates whether a model trained on multiple participants can generalize to a participant that was not seen during training.

This is more challenging and more relevant for assessing cross-subject robustness.

### Observation-time analysis

Observation-time analysis evaluates how much EEG data are needed to obtain stable decoding performance.

This is important for real-world or mobile monitoring scenarios, where long recordings may not always be available.

### Time-on-task analysis

Time-on-task analysis investigates whether EEG features change across task duration.

This is useful for studying temporal drift, fatigue-related effects, arousal changes and non-stationarity in EEG workload markers.

### Cross-paradigm transfer

Cross-paradigm transfer evaluates whether a model trained on one workload manipulation can generalize to another.

This helps distinguish task-specific effects from more general EEG workload signatures.

---

## Synthetic data examples

Some scripts generate synthetic EEG-like signals or synthetic feature tables when no input file is provided.

Synthetic data are used only to make the scripts executable and to demonstrate the structure of the workflow.

They are not intended to reproduce the original experimental results.

---

## Expected real-data structure

When using real data, the expected structure depends on the analysis module.

For spectral feature extraction, input data should include EEG channels:

| Column | Electrode |
| ------ | --------- |
| Ch01   | Fz        |
| Ch02   | C3        |
| Ch03   | Cz        |
| Ch04   | C4        |
| Ch05   | Pz        |
| Ch06   | PO7       |
| Ch07   | POz       |
| Ch08   | PO8       |

Additional marker columns may be present, but should be handled separately from EEG channels.

For classification and temporal analyses, the expected input is a feature table with:

* one row per EEG window or segment;
* numeric EEG feature columns;
* condition or workload labels;
* subject identifiers when subject-level validation is required;
* timing or segment information when temporal analyses are performed.

---

## Limitations of this repository

This repository is not a full reproduction package of the thesis results.

It does not include the original dataset and therefore cannot reproduce the exact numerical results of the thesis.

Its purpose is to document the experimental and computational workflow in a transparent, privacy-preserving and portfolio-oriented form.
