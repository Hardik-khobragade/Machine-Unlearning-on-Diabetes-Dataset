# =======================================================
# TEAM 3 - MODEL TESTING AND SELECTION PIPELINE
# =======================================================
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# CONFIGURATION
DATA_PATH = "sisa_data/shard_0_split_0.csv"  # use any one split
TARGET_COLUMN = "Outcome"  # change if different

def load_data():
    df = pd.read_csv(DATA_PATH)
    if "user_id" in df.columns:
        df = df.drop(columns=["user_id", "index"], errors="ignore")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y

def evaluate_models(X, y):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        results[name] = scores.mean()
        print(f"{name:<20} | Accuracy: {scores.mean():.4f}")

    best_model = max(results, key=results.get)
    print("\nBest Model:", best_model)
    return best_model, results[best_model]

def main():
    X, y = load_data()
    best_model, acc = evaluate_models(X, y)
    print(f"\n✅ Best model is '{best_model}' with accuracy {acc:.4f}")

if __name__ == "__main__":
    main()
