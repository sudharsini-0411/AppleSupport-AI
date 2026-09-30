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
    f1_score,
    classification_report
)


# ======================================
# FILE PATHS
# ======================================

train_file = "data/processed/apple_support_labeled_v2.csv"
golden_file = "data/processed/golden_evaluation_final.csv"


# ======================================
# LOAD DATA
# ======================================

train_df = pd.read_csv(train_file)
golden_df = pd.read_csv(golden_file)

print("Training rows:", len(train_df))
print("Golden rows:", len(golden_df))


# ======================================
# TRAINING DATA
# ======================================

X_train = train_df["customer_message"].fillna("")
y_train = train_df["intent"]


# ======================================
# GOLDEN EVALUATION DATA
# ======================================

X_test = golden_df["customer_message"].fillna("")

# IMPORTANT:
# Use corrected_intent as the trusted ground truth
y_test = golden_df["corrected_intent"]


# ======================================
# TF-IDF VECTORIZATION
# ======================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)


# ======================================
# LOGISTIC REGRESSION
# ======================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# ======================================
# PREDICTION
# ======================================

y_pred = model.predict(X_test_tfidf)


# ======================================
# FINAL RESULTS
# ======================================

print("\n======================================")
print("FINAL TF-IDF BASELINE")
print("======================================")

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")

print(
    f"Precision: "
    f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}"
)

print(
    f"Recall   : "
    f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}"
)

print(
    f"F1 Score : "
    f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}"
)


# ======================================
# MACRO F1
# ======================================

print(
    f"Macro F1 : "
    f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}"
)


# ======================================
# CLASSIFICATION REPORT
# ======================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ======================================
# SAMPLE PREDICTIONS
# ======================================

print("\n======================================")
print("SAMPLE PREDICTIONS")
print("======================================")

results = pd.DataFrame({
    "customer_message": golden_df["customer_message"],
    "actual_intent": golden_df["corrected_intent"],
    "predicted_intent": y_pred
})

print(results.head(20).to_string(index=False))