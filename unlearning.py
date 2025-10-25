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
STUDENT_MODEL_PATH = os.path.join(MODEL_DIR, "student_model.pkl")
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
    
    def predict_proba(self, X):
        preds = [model.predict_proba(X) for model in self.models]
        avg_pred = sum(preds) / len(preds)
        return avg_pred

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

# -------------------- STUDENT MODEL TRAINING --------------------
def train_student_model(teacher_model, shard_paths):
    """Train student model using hard labels from teacher ensemble"""
    print("\n Training Student Model using Hard Labels...")
    
    # Collect all data and teacher predictions
    X_all = []
    y_hard_all = []
    
    for path in shard_paths:
        X, y = load_dataset(path)
        # Get hard predictions from teacher (predicted class labels)
        y_hard = teacher_model.predict(X)
        X_all.append(X)
        y_hard_all.append(y_hard)
    
    X_combined = pd.concat(X_all, axis=0, ignore_index=True)
    y_hard_combined = np.concatenate(y_hard_all)
    
    # Train student model on hard labels
    student_model = LogisticRegression(max_iter=500)
    student_model.fit(X_combined, y_hard_combined)
    
    # Save student model
    save_model(student_model, STUDENT_MODEL_PATH)
    print(f"✅ Student model saved to {STUDENT_MODEL_PATH}")
    
    return student_model

def evaluate_student_model(student_model, shard_paths):
    """Evaluate student model accuracy"""
    print("\n Calculating Student Model accuracy...")
    accs = []
    for path in shard_paths:
        X, y = load_dataset(path)
        preds = student_model.predict(X)
        acc = accuracy_score(y, preds)
        accs.append(acc)
    avg_acc = np.mean(accs)
    print(f"Student Model Accuracy: {avg_acc:.4f}")
    return avg_acc

# -------------------- UNLEARNING FUNCTION --------------------
def unlearn_user(user_id, model_paths, shard_paths):
    user_id_key = str(int(float(user_id)))  # Ensure consistent string key
    if not os.path.exists(MAPPING_FILE):
        print("❌ No user mapping file found.")
        return []

    with open(MAPPING_FILE, "r") as f:
        mapping = json.load(f)

    if user_id_key not in mapping:
        print(f"❌ User {user_id_key} not found in mapping. Nothing to unlearn.")
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

            # Retrain model for this shard
            if len(df) > 0:
                X, y = load_dataset(csv_path)
                model = LogisticRegression(max_iter=500)
                model.fit(X, y)
                model_file = os.path.join(MODEL_DIR, f"model_shard_{shard}_split_{split}.pkl")
                save_model(model, model_file)
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
    for path in shard_paths:
        basename = os.path.basename(path)
        # Extract shard and split numbers from filename like "shard_0_split_0.csv"
        parts = basename.replace('.csv', '').split('_')
        shard_num = parts[1]
        split_num = parts[3]
        
        print(f"Training on {basename}")
        X, y = load_dataset(path)
        model = LogisticRegression(max_iter=500)
        model.fit(X, y)
        model_file = os.path.join(MODEL_DIR, f"model_shard_{shard_num}_split_{split_num}.pkl")
        save_model(model, model_file)
        model_paths.append(model_file)

    # Initial SISA ensemble accuracy
    print("\n Calculating initial SISA accuracy...")
    combined_model = combine_models(model_paths)
    accs = []
    for path in shard_paths:
        X, y = load_dataset(path)
        preds = combined_model.predict(X)
        acc = accuracy_score(y, preds)
        accs.append(acc)
    initial_sisa_acc = np.mean(accs)
    print(f"Initial SISA Accuracy: {initial_sisa_acc:.4f}")

    # Get user input to unlearn
    user_to_unlearn = input("\n👤 Enter user_id to unlearn: ")
    affected = unlearn_user(user_to_unlearn, model_paths, shard_paths)
    
    if affected:
        print(f"📝 Affected shards/splits: {affected}")
        
        # Reload updated ensemble after unlearning
        print("\n🔄 Reloading updated SISA ensemble...")
        combined_model = combine_models(model_paths)
        
        # Train student model ONLY after unlearning
        student_model = train_student_model(combined_model, shard_paths)
        student_acc = evaluate_student_model(student_model, shard_paths)
        
        # Comparison
        print("\n" + "="*60)
        print("📊 ACCURACY COMPARISON")
        print("="*60)
        print(f"   Initial SISA Accuracy:           {initial_sisa_acc:.4f}")
        print(f"   Student Model (after unlearning): {student_acc:.4f}")
        print(f"   Difference:                       {student_acc - initial_sisa_acc:+.4f}")
        print("="*60)
    else:
        print("\n⚠️  No unlearning performed. Student model not trained.")

    # Remaining users
    remaining_users = get_all_user_ids()
    print(f"\n📋 Remaining user IDs in mapping: {remaining_users}")

    print("\n✅ Pipeline complete. All models saved in:", MODEL_DIR)

if __name__ == "__main__":
    main()


