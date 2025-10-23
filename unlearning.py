import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -------------------- CONFIG --------------------
SISA_DIR = "sisa_data/"
MODEL_DIR = os.path.join(SISA_DIR, "models/")
MAPPING_FILE = os.path.join(SISA_DIR, "user_mapping.json")
TARGET_COLUMN = "Outcome"

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------- COMBINED MODEL --------------------
class CombinedModel:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [model.predict_proba(X) for model in self.models]
        avg_pred = sum(preds) / len(preds)
        return avg_pred.argmax(axis=1)

# -------------------- HELPERS --------------------
def load_dataset(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["user_id", "index"], errors="ignore")
    if df[TARGET_COLUMN].dtype == "object":
        df[TARGET_COLUMN] = LabelEncoder().fit_transform(df[TARGET_COLUMN])
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def combine_models(model_paths):
    models = [load_model(p) for p in model_paths]
    return CombinedModel(models)

def get_all_user_ids():
    """Return all user IDs still present in the mapping"""
    if not os.path.exists(MAPPING_FILE):
        return []
    with open(MAPPING_FILE, "r") as f:
        mapping = json.load(f)
    return sorted(int(uid) for uid in mapping.keys())

# -------------------- UNLEARNING FUNCTION --------------------
def unlearn_user(user_id, model_paths):
    user_id_key = str(int(float(user_id)))  # Ensure consistent string key
    if not os.path.exists(MAPPING_FILE):
        print("❌ No user mapping file found.")
        return []

    with open(MAPPING_FILE, "r") as f:
        mapping = json.load(f)

    if user_id_key not in mapping:
        print(f"User {user_id_key} not found in mapping. Nothing to unlearn.")
        return []

    user_info = mapping[user_id_key]
    locations = user_info.get("locations", [])

    affected_shards = []

    for loc in locations:
        shard = loc["shard"]
        split = loc["split"]
        rows_to_remove = loc["rows"]
        csv_path = os.path.join(SISA_DIR, f"shard_{shard}_split_{split}.csv")

        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        # Only drop if rows exist
        rows_in_df = [r for r in rows_to_remove if r in df.index]
        if rows_in_df:
            df = df.drop(rows_in_df, axis=0)
            df.to_csv(csv_path, index=False)

            # Retrain model for this shard (assuming one model per shard)
            if len(df) > 0:
                X, y = load_dataset(csv_path)
                model = LogisticRegression(max_iter=500)
                model.fit(X, y)
                save_model(model, model_paths[shard])
                affected_shards.append(f"shard_{shard}_split_{split}")

    # Remove user from mapping
    del mapping[user_id_key]
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=4)

    print(f"✅ User {user_id_key} removed from all listed shards/splits.")
    return affected_shards

# -------------------- MAIN PIPELINE --------------------
def main():
    # Get all shards/splits
    shard_paths = sorted(glob.glob(os.path.join(SISA_DIR, "shard_*_split_*.csv")))
    if not shard_paths:
        print("❌ No shard files found in sisa_data/.")
        return

    # Train all shards initially
    model_paths = []
    print("Training all shards with LogisticRegression...\n")
    for i, path in enumerate(shard_paths):
        print(f"[{i+1}/{len(shard_paths)}] Training on {os.path.basename(path)}")
        X, y = load_dataset(path)
        model = LogisticRegression(max_iter=500)
        model.fit(X, y)
        model_file = os.path.join(MODEL_DIR, f"model_shard_{i}.pkl")
        save_model(model, model_file)
        model_paths.append(model_file)

    # Initial ensemble accuracy
    print("\n🔹 Calculating initial SISA accuracy...")
    combined_model = combine_models(model_paths)
    accs = []
    for path in shard_paths:
        X, y = load_dataset(path)
        preds = combined_model.predict(X)
        acc = accuracy_score(y, preds)
        accs.append(acc)
    print(f"Initial SISA Accuracy: {np.mean(accs):.4f}")

    # Get user input to unlearn
    user_to_unlearn = input("\nEnter user_id to unlearn: ")
    affected = unlearn_user(user_to_unlearn, model_paths)
    if affected:
        print(f"Affected shards/splits: {affected}")

    # Remaining users
    remaining_users = get_all_user_ids()
    print(f"\nRemaining user IDs in mapping: {remaining_users}")

    print("\n✅ Pipeline complete. All models saved in:", MODEL_DIR)

if __name__ == "__main__":
    main()



# Improvements and some changes have been made!!!
