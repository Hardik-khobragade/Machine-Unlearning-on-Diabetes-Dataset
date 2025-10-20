# 🧠 Machine Unlearning on Diabetes Dataset

This project demonstrates **machine unlearning** using the **Diabetes dataset**.
The main goal is to ensure **data privacy** by enabling deletion of specific user data (User ID) upon request, while maintaining model performance through SISA (Sharded, Isolated, Sliced, and Aggregated) training.

---

## ⚙️ 1. Environment Setup

### Prerequisites

* Python **3.10**
* Recommended: `virtualenv` for isolation

```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate        # for Linux/macOS
venv\Scripts\activate           # for Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🧩 2. Data Sharding and Splitting

Run the notebook `Data_sharding_splitting.ipynb`.

This will:

* Split the original Diabetes dataset into **3 shards**, each containing **4 splits**.
* Generate a **user mapping file** for each user ID.
* Create a new directory:

```
sisa_data/
│
├── shard_0_split_0.csv
├── shard_0_split_1.csv
├── ...
└── user_mapping.json
```

---

## 🔒 3. User Unlearning (Data Deletion)

Rename the file `sisa_unlearn.py` to `Remove_user_id.py` and execute:

```bash
python Remove_user_id.py
```

### Functionality:

* Prompts for a `User ID`.
* Locates and removes all occurrences of that user from affected shard-split files.
* Saves the modified splits to:

  ```
  sisa_data/unlearned_splits/
  ```
* Ensures complete isolation of deleted user data for privacy compliance.

---



## 🧩 4. Train All Shards with Logostic regression Model

Run:

```bash
python train_all_shards_with_best.py
```

This step:

* Reads the best model from `best_model.json`
* Trains that model on all shards and splits
* Saves trained models in:

  ```
  trained_models/
  ```
* Combines all trained models into a final unified model:

  ```
  combined_model/combined_model.pkl
  ```

---

## 📁 Project Structure

```
.
├── best_model_selector.py
├── best_model.json
├── Data_sharding_splitting.ipynb
├── diabetes_with_users_reordered.csv
├── model_selection.py
├── Remove_user_id.py
├── train_all_shards_with_best.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 👥 Team

**TEAM 2 - SISA Unlearning Implementation**

This project ensures user-level data privacy and compliance with unlearning protocols in machine learning workflows.
