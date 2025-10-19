# =======================================================
# TEAM 3A - BEST MODEL SELECTION
# =======================================================
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------- CONFIG --------------------
DATA_PATH = "sisa_data/shard_0_split_0.csv"  # sample file
TARGET_COLUMN = "Outcome"                    # change if needed
OUTPUT_FILE = "best_model.json"

# -------------------- MAIN --------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["user_id", "index"], errors="ignore")
    if df[TARGET_COLUMN].dtype == "object":
        df[TARGET_COLUMN] = LabelEncoder().fit_transform(df[TARGET_COLUMN])
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y

def evaluate_models(X, y):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    best_model, best_acc = None, 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Evaluating models...\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name:<20} | Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_model, best_acc = name, acc

    print(f"\n✅ Best model selected: {best_model} (Acc: {best_acc:.4f})")

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"best_model": best_model, "accuracy": best_acc}, f, indent=4)
    print(f"Saved result to {OUTPUT_FILE}")

def main():
    X, y = load_data()
    evaluate_models(X, y)

if __name__ == "__main__":
    main()

