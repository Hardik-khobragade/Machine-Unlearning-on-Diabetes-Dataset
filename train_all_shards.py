# =======================================================
# TEAM 2 - TRAIN ALL SHARDS AND IMPLEMENT UNLEARNING
# =======================================================
import os
import glob
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------- CONFIG --------------------
SISA_DIR = "sisa_data/"
MODEL_DIR = "trained_models/"
BEST_MODEL_FILE = "best_model.json"
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
    return X, y, df

def get_model(name="LogisticRegression"):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=500)
    else:
        raise ValueError("Only LogisticRegression is implemented.")

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def combine_models(model_paths):
    models = [load_model(p) for p in model_paths]
    combined_model = CombinedModel(models)
    combined_model_path = os.path.join(MODEL_DIR, "combined_model.pkl")
    save_model(combined_model, combined_model_path)
    return combined_model

# -------------------- UNLEARNING FUNCTION --------------------
def unlearn_user(user_id, shard_paths, model_paths):
    print(f"\n🔹 Unlearning user_id: {user_id}")
    affected_shards = []
    for i, path in enumerate(shard_paths):
        df = pd.read_csv(path)
        if "user_id" in df.columns and user_id in df["user_id"].values:
            print(f" - Updating shard: {os.path.basename(path)}")
            df = df[df["user_id"] != user_id]  # remove user data
            df.to_csv(path, index=False)

            # retrain model on updated shard
            X, y, _ = load_dataset(path)
            model = get_model()
            model.fit(X, y)
            save_model(model, model_paths[i])
            affected_shards.append(i)
    print(f"✅ Unlearning done. Updated shards: {affected_shards}\n")
    return affected_shards

# -------------------- MAIN PIPELINE --------------------
def main():
    if not os.path.exists(BEST_MODEL_FILE):
        print("❌ Run best_model_selector.py first to determine best model.")
        return

    with open(BEST_MODEL_FILE, "r") as f:
        best_info = json.load(f)
        model_name = best_info.get("best_model", "LogisticRegression")
    print(f"Training all shards using model: {model_name}\n")

    shard_paths = sorted(glob.glob(os.path.join(SISA_DIR, "shard_*_split_*.csv")))
    if not shard_paths:
        print("❌ No shard files found in sisa_data/.")
        return

    model_paths = []

    # Train all shards
    for i, path in enumerate(shard_paths):
        print("="*50)
        print(f"[{i+1}/{len(shard_paths)}] Training on {os.path.basename(path)}")
        X, y, _ = load_dataset(path)
        model = get_model(model_name)
        model.fit(X, y)
        model_file = os.path.join(MODEL_DIR, f"{model_name}_shard_{i}.pkl")
        save_model(model, model_file)
        model_paths.append(model_file)

    # Ensemble before unlearning
    print("\n🔹 Calculating ensemble accuracy BEFORE unlearning...")
    combined_model = combine_models(model_paths)
    accs = []
    for path in shard_paths:
        X, y, _ = load_dataset(path)
        preds = combined_model.predict(X)
        acc = accuracy_score(y, preds)
        accs.append(acc)
    print(f"✅ Average ensemble accuracy: {np.mean(accs):.4f}")

    # Example: Unlearning a specific user
    user_to_unlearn = 123  # Replace with actual user_id
    unlearn_user(user_to_unlearn, shard_paths, model_paths)

    # Ensemble after unlearning
    print("🔹 Calculating ensemble accuracy AFTER unlearning...")
    combined_model = combine_models(model_paths)
    accs = []
    for path in shard_paths:
        X, y, _ = load_dataset(path)
        preds = combined_model.predict(X)
        acc = accuracy_score(y, preds)
        accs.append(acc)
    print(f"✅ Average ensemble accuracy after unlearning: {np.mean(accs):.4f}")

    print("\n✅ Pipeline complete. All models updated in-place in:", MODEL_DIR)

if __name__ == "__main__":
    main()
