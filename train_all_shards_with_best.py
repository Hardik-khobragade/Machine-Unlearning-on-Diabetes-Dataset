# =======================================================
# TEAM 3B - TRAIN ALL SHARDS USING BEST MODEL
# =======================================================
import os
import joblib
import json
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# -------------------- CONFIG --------------------
SISA_DIR = "sisa_data/"
MODEL_DIR = "trained_models/"
BEST_MODEL_FILE = "best_model.json"
TARGET_COLUMN = "Outcome"
COMBINED_MODEL_PATH = os.path.join(MODEL_DIR, "final_combined_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

class CombinedModel:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        # Example: simple averaging of predictions
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

def get_model(name):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }
    return models[name]

def combine_models(model_paths):
    models = [joblib.load(path) for path in model_paths]
    combined_model = CombinedModel(models)

    os.makedirs("combined_model", exist_ok=True)
    combined_model_path = "combined_model/combined_model.pkl"

    # Use pickle to save combined model
    with open(combined_model_path, "wb") as f:
        pickle.dump(combined_model, f)

    print(f"✅ Combined model saved as {combined_model_path}")

# -------------------- MAIN PIPELINE --------------------
def main():
    if not os.path.exists(BEST_MODEL_FILE):
        print("❌ Run best_model_selector.py first to determine best model.")
        return

    with open(BEST_MODEL_FILE, "r") as f:
        best_info = json.load(f)
        model_name = best_info["best_model"]
        print(f"Training all shards using best model: {model_name}\n")

    shard_paths = sorted(glob.glob(os.path.join(SISA_DIR, "shard_*_split_*.csv")))
    if not shard_paths:
        print("No shard files found in sisa_data/.")
        return

    model_paths = []
    for i, path in enumerate(shard_paths):
        print("="*60)
        print(f"[{i+1}/{len(shard_paths)}] Training on {os.path.basename(path)}")

        X, y = load_dataset(path)
        model = get_model(model_name)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"Accuracy: {acc:.4f}")
        model_file = os.path.join(MODEL_DIR, f"{model_name}_shard_{i}.pkl")
        pickle.dump(model, open(model_file, "wb"))
        model_paths.append(model_file)

    print("\nCombining all trained models...")
    combine_models(model_paths)
    print("\n✅ Training complete. All models stored in:", MODEL_DIR)

if __name__ == "__main__":
    main()
