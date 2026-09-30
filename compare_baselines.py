import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sentence_transformers import SentenceTransformer

# -----------------------------------------------
# 1. Load data
# -----------------------------------------------

train_df = pd.read_csv("data/processed/apple_support_balanced_v2.csv")
golden_df = pd.read_csv("data/processed/golden_evaluation_final.csv")

X_train = train_df["customer_message"].fillna("")
y_train = train_df["intent"]

X_test = golden_df["customer_message"].fillna("")
y_test = golden_df["corrected_intent"]   # trusted ground truth

# -----------------------------------------------
# 2. TF-IDF + Logistic Regression
# -----------------------------------------------

print("Running TF-IDF + Logistic Regression...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

tfidf_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
tfidf_clf.fit(X_train_tfidf, y_train)
y_pred_tfidf = tfidf_clf.predict(X_test_tfidf)

tfidf_scores = {
    "Accuracy"          : accuracy_score(y_test, y_pred_tfidf),
    "Precision (weighted)": precision_score(y_test, y_pred_tfidf, average="weighted", zero_division=0),
    "Recall (weighted)" : recall_score(y_test, y_pred_tfidf, average="weighted", zero_division=0),
    "F1 (weighted)"     : f1_score(y_test, y_pred_tfidf, average="weighted", zero_division=0),
    "F1 (macro)"        : f1_score(y_test, y_pred_tfidf, average="macro", zero_division=0),
}

# -----------------------------------------------
# 3. Sentence Transformer + Logistic Regression
# -----------------------------------------------

print("Running Sentence Transformer + Logistic Regression...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

X_train_emb = embedder.encode(
    X_train.tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)
X_test_emb = embedder.encode(
    X_test.tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

st_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
st_clf.fit(X_train_emb, y_train)
y_pred_st = st_clf.predict(X_test_emb)

st_scores = {
    "Accuracy"          : accuracy_score(y_test, y_pred_st),
    "Precision (weighted)": precision_score(y_test, y_pred_st, average="weighted", zero_division=0),
    "Recall (weighted)" : recall_score(y_test, y_pred_st, average="weighted", zero_division=0),
    "F1 (weighted)"     : f1_score(y_test, y_pred_st, average="weighted", zero_division=0),
    "F1 (macro)"        : f1_score(y_test, y_pred_st, average="macro", zero_division=0),
}

# -----------------------------------------------
# 4. Side-by-side comparison table
# -----------------------------------------------

print("\n")
print("=" * 62)
print("  BASELINE COMPARISON — Corrected Golden Set (200 examples)")
print("=" * 62)
print(f"{'Metric':<25} {'TF-IDF + LR':>15} {'ST + LR':>15}")
print("-" * 62)

for metric in tfidf_scores:
    t = tfidf_scores[metric]
    s = st_scores[metric]
    winner = " ◄" if s > t else ("" if s == t else "")
    print(f"{metric:<25} {t:>15.4f} {s:>15.4f}{winner}")

print("=" * 62)
print("◄ = Sentence Transformer wins on this metric")

# -----------------------------------------------
# 5. Save comparison to CSV
# -----------------------------------------------

comparison_df = pd.DataFrame({
    "Metric"       : list(tfidf_scores.keys()),
    "TF-IDF + LR"  : list(tfidf_scores.values()),
    "ST + LR"      : list(st_scores.values()),
})
comparison_df["Improvement"] = comparison_df["ST + LR"] - comparison_df["TF-IDF + LR"]

comparison_df.to_csv(
    "data/processed/baseline_comparison.csv",
    index=False
)

print("\nComparison saved: data/processed/baseline_comparison.csv")

# -----------------------------------------------
# 6. Per-class F1 comparison
# -----------------------------------------------

from sklearn.metrics import classification_report
import numpy as np

tfidf_report = classification_report(y_test, y_pred_tfidf, output_dict=True, zero_division=0)
st_report    = classification_report(y_test, y_pred_st,    output_dict=True, zero_division=0)

intents = sorted(golden_df["corrected_intent"].unique())

print("\n")
print("=" * 70)
print("  PER-CLASS F1 COMPARISON")
print("=" * 70)
print(f"{'Intent':<22} {'TF-IDF F1':>12} {'ST F1':>12} {'Delta':>10}")
print("-" * 70)

for intent in intents:
    t_f1 = tfidf_report.get(intent, {}).get("f1-score", 0.0)
    s_f1 = st_report.get(intent, {}).get("f1-score", 0.0)
    delta = s_f1 - t_f1
    arrow = " ▲" if delta > 0.01 else (" ▼" if delta < -0.01 else "  ~")
    print(f"{intent:<22} {t_f1:>12.4f} {s_f1:>12.4f} {delta:>+10.4f}{arrow}")

print("=" * 70)
print("▲ = ST improved  ▼ = TF-IDF was better  ~ = similar")
