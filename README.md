# Machine Unlearning 

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SISA%20Unlearning-green?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model%20Training-F7931E?style=for-the-badge&logo=scikitlearn)
![NumPy](https://img.shields.io/badge/NumPy-Enabled-blue?style=for-the-badge&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas)
![Repo Size](https://img.shields.io/github/repo-size/Hardik-khobragade/Machine-Unlearning-on-Diabetes-Dataset?style=for-the-badge)
![Code Count](https://img.shields.io/github/languages/count/Hardik-khobragade/Machine-Unlearning-on-Diabetes-Dataset?style=for-the-badge)
![Top Language](https://img.shields.io/github/languages/top/Hardik-khobragade/Machine-Unlearning-on-Diabetes-Dataset?style=for-the-badge)
![GitHub Stars](https://img.shields.io/github/stars/Hardik-khobragade/Machine-Unlearning-on-Diabetes-Dataset?style=for-the-badge&logo=github)
![GitHub Forks](https://img.shields.io/github/forks/Hardik-khobragade/Machine-Unlearning-on-Diabetes-Dataset?style=for-the-badge&logo=github)
![GitHub Issues](https://img.shields.io/github/issues/Hardik-khobragade/Machine-Unlearning-on-Diabetes-Dataset?style=for-the-badge&logo=github)
![GitHub PRs](https://img.shields.io/github/issues-pr/Hardik-khobragade/Machine-Unlearning-on-Diabetes-Dataset?style=for-the-badge&logo=github)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Last Commit](https://img.shields.io/github/last-commit/Hardik-khobragade/Machine-Unlearning-on-Diabetes-Dataset?style=for-the-badge&logo=git)
![Contributors](https://img.shields.io/github/contributors/Hardik-khobragade/Machine-Unlearning-on-Diabetes-Dataset?style=for-the-badge&logo=github)
![Commit Activity](https://img.shields.io/github/commit-activity/m/Hardik-khobragade/Machine-Unlearning-on-Diabetes-Dataset?style=for-the-badge)

---
A Complete Implementation of SISA Sharding, User Mapping, Unlearning, and Student Model Distillation

---

## Overview

This project demonstrates a full **Machine Unlearning pipeline** implementing the **SISA (Sharded, Isolated, Slice-based Aggregation)** technique on the PIMA Diabetes Dataset.
It includes:

* Data preprocessing and user ID assignment
* SISA sharding and splitting
* User-to-data mapping generation
* Per-shard model training
* Machine unlearning for specific users
* Hard-label student model distillation
* Post-unlearning accuracy evaluation

The system ensures traceability, verifiable unlearning, and efficient retraining.

---

## Project Structure

```
Machine-Unlearning-on-Diabetes-Dataset/
│
├── SISA_processing.py        # Team 1: Preprocessing, sharding, mapping
├── unlearning.py             # Team 3: Unlearning + model retraining + distillation
│
├── diabetes_with_users_reordered.csv  # Input dataset, This dataset is for demonstration only, This project is dataset independent
│
├── sisa_data/                # Generated automatically
│   ├── shard_0_split_0.csv
│   ├── shard_0_split_1.csv
│   ├── ...
│   ├── models/
│   │   ├── model_shard_0_split_0.pkl
│   │   ├── student_model.pkl
│   │   └── ...
│   ├── user_mapping.json
│   └── ...
│
└── README.md
```

---

## Core Components

### 1. Dataset Preprocessing (SISA_processing.py – Cell 1)

* Detects or creates a `user_id` column.
* Handles missing values.
* Encodes non-numeric features.
* Adds an `index` column to preserve original row identity.
* Prepares the dataset for deterministic user tracking.

### 2. SISA Sharding & Detailed User Mapping (Cell 2)

Creates:

* `num_shards` shards
* `splits_per_shard` splits per shard
* A full **user-to-shard/split mapping**, stored in:

  ```
  sisa_data/user_mapping.json
  ```

Each user entry includes:

* All original row indices
* All related (shard, split) locations
* Exact row-level tracking for each split

This mapping enables **auditable** and **precise unlearning**.

### 3. Saving SISA Outputs

All shards and user mapping files are saved inside:

```
sisa_data/
```

---

## Machine Unlearning Pipeline (unlearning.py)

### 1. Initial Shard Training

Each shard/split CSV is used to train a separate Logistic Regression model.
Models are saved under:

```
sisa_data/models/model_shard_X_split_Y.pkl
```

### 2. SISA Ensemble Construction

A `CombinedModel` class averages probabilities across all trained shard models.

### 3. User Unlearning

Given a `user_id`, the pipeline:

* Reads `user_mapping.json`
* Locates affected shard/split files
* Removes all rows associated with that user
* Retrains only the impacted shards
* Updates the user mapping

This ensures efficient and isolated unlearning.

### 4. Student Model Distillation

After unlearning:

* The SISA ensemble generates **hard labels**
* A lightweight Logistic Regression student model is trained
* The student model is saved to:

```
sisa_data/models/student_model.pkl
```

### 5. Accuracy Evaluation

The script compares:

* Initial SISA model accuracy
* Post-unlearning student model accuracy

This helps evaluate the stability of training versus the impact of unlearning.

---

## How to Run the Pipeline

### Step 1: Generate SISA Shards and Mapping

```
python SISA_processing.py
```

This creates:

* Shards & splits
* User mapping JSON
* Preprocessed dataset logs

### Step 2: Run the Unlearning System

```
python unlearning.py
```

You will be prompted:

```
Enter user_id to unlearn:
```

After execution, the script will output:

* Affected shards
* Updated ensemble accuracy
* Student model accuracy
* Remaining users in mapping

---


## Requirements

Install necessary dependencies using:

```
pip install -r requirements.txt
```

Typical libraries used:

```
numpy
pandas
scikit-learn
joblib
```

---

## Output Files Summary

| File / Folder                      | Description                               |
| ---------------------------------- | ----------------------------------------- |
| `sisa_data/shard_X_split_Y.csv`    | SISA dataset partitions                   |
| `sisa_data/user_mapping.json`      | Complete user-to-data mapping             |
| `models/model_shard_X_split_Y.pkl` | Trained shard models                      |
| `models/student_model.pkl`         | Post-unlearning lightweight student model |

---

## Key Features

* Full SISA framework implementation
* Deterministic user-level tracking
* Efficient per-shard retraining
* Complete machine unlearning for any user
* Hard-label student model distillation
* Accuracy before vs. after unlearning
* Modular, transparent, and auditable pipeline

---

## Future Improvements

* Support for Soft-Label distillation
* GPU-accelerated training
* Differential privacy integration
* API endpoint for user unlearning requests
* Dashboard for shard-level training metrics

---

## License

This project is released under the MIT License.

---

## Contact

For queries or collaboration, reach out to the contributors.
