"""
train_models.py
----------------
Loads data/sample_data.csv (swap in the real merged LIAR+ISOT+Kaggle dataset
for the actual submission), cleans the text, builds TF-IDF features, trains
several classic ML models, evaluates them, and saves:
  - models/vectorizer.pkl   (fitted TF-IDF vectorizer)
  - models/best_model.pkl   (best-performing model)
  - models/metrics.json     (comparison table used in the report / demo)

Run:
    python src/train_models.py
"""
import json
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocess import clean_text

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def build_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "SVM (Linear)": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42)
    return models


def main():
    print("Loading and cleaning data...")
    df = load_data()
    print(f"Total samples: {len(df)}  (real={sum(df.label == 0)}, fake={sum(df.label == 1)})")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = {}
    best_model_name, best_model, best_f1 = None, None, -1

    for name, model in build_models().items():
        print(f"Training {name}...")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        }
        results[name] = metrics
        print(f"  -> {metrics}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name
            best_model = model

    print(f"\nBest model: {best_model_name} (F1={best_f1})")

    with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_model_name, "results": results}, f, indent=2)

    print(f"\nSaved vectorizer + best model + metrics.json to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
