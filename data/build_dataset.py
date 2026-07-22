"""
build_dataset.py
-----------------
Merges the raw source datasets in data/ into data/sample_data.csv
(text,label) used by src/train_models.py.

Sources:
  - True.csv / Fake.csv        ISOT dataset (title, text, subject, date)
  - GlobalFakeNews_Research2026_v1.csv   (title, full_text, ..., label)

Run:
    python data/build_dataset.py
"""
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

true_df = pd.read_csv(os.path.join(BASE_DIR, "True.csv"))
fake_df = pd.read_csv(os.path.join(BASE_DIR, "Fake.csv"))
global_df = pd.read_csv(os.path.join(BASE_DIR, "GlobalFakeNews_Research2026_v1.csv"))

true_df["text"] = (true_df["title"].fillna("") + " " + true_df["text"].fillna("")).str.strip()
true_df["label"] = 0

fake_df["text"] = (fake_df["title"].fillna("") + " " + fake_df["text"].fillna("")).str.strip()
fake_df["label"] = 1

global_df["text"] = (global_df["title"].fillna("") + " " + global_df["full_text"].fillna("")).str.strip()
global_df = global_df[["text", "label"]]

combined = pd.concat(
    [true_df[["text", "label"]], fake_df[["text", "label"]], global_df[["text", "label"]]],
    ignore_index=True,
)

combined = combined.dropna(subset=["text", "label"])
combined = combined[combined["text"].str.len() > 20]
combined = combined.drop_duplicates(subset=["text"])
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

out_path = os.path.join(BASE_DIR, "sample_data.csv")
combined.to_csv(out_path, index=False)

print(f"Total samples: {len(combined)}")
print(combined["label"].value_counts())
print(f"Saved to {out_path}")
