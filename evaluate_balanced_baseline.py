import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

train_file = "data/processed/apple_support_balanced.csv"
golden_file = "data/processed/golden_evaluation_set.csv"

# Load data
train_df = pd.read_csv(train_file)
golden_df = pd.read_csv(golden_file)

print("Training rows:", len(train_df))
print("Golden rows:", len(golden_df))

# Training data
X_train = train_df["customer_message"].fillna("")
y_train = train_df["intent"]

# Golden data
X_golden = golden_df["customer_message"].fillna("")
y_golden = golden_df["actual_intent"]

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_golden_tfidf = vectorizer.transform(X_golden)

# Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

# Prediction
y_pred = model.predict(X_golden_tfidf)

# Metrics
accuracy = accuracy_score(y_golden, y_pred)

precision = precision_score(
    y_golden,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_golden,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_golden,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n========================================")
print("BALANCED TF-IDF BASELINE")
print("========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_golden,
        y_pred,
        zero_division=0
    )
)

print("\nEvaluation completed!")