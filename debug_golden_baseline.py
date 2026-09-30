import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==============================
# FILES
# ==============================

train_file = "data/processed/apple_support_labeled.csv"
golden_file = "data/processed/golden_evaluation_set.csv"


# ==============================
# LOAD DATA
# ==============================

train_df = pd.read_csv(train_file)
golden_df = pd.read_csv(golden_file)

print("Training rows:", len(train_df))
print("Golden rows:", len(golden_df))


# ==============================
# TRAINING DATA
# ==============================

X_train = train_df["customer_message"].fillna("")
y_train = train_df["intent"]


# ==============================
# GOLDEN DATA
# ==============================

X_golden = golden_df["customer_message"].fillna("")
y_golden = golden_df["actual_intent"]


# ==============================
# TF-IDF
# ==============================

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_golden_tfidf = vectorizer.transform(X_golden)


# ==============================
# LOGISTIC REGRESSION
# ==============================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)


# ==============================
# PREDICTION
# ==============================

y_pred = model.predict(X_golden_tfidf)


# ==============================
# METRICS
# ==============================

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


# ==============================
# MAIN RESULTS
# ==============================

print("\n========================================")
print("GOLDEN SET - TF-IDF BASELINE")
print("========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ==============================
# CLASSIFICATION REPORT
# ==============================

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


# ==============================
# CONFUSION MATRIX
# ==============================

labels = sorted(
    set(y_golden) | set(y_pred)
)

cm = confusion_matrix(
    y_golden,
    y_pred,
    labels=labels
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print("Labels:")
print(labels)

print("\nMatrix:")
print(cm)


# ==============================
# WRONG PREDICTIONS
# ==============================

wrong = golden_df.copy()

wrong["predicted_intent"] = y_pred

wrong = wrong[
    wrong["actual_intent"] != wrong["predicted_intent"]
]

print("\n========================================")
print("WRONG PREDICTIONS")
print("========================================")

print("Total wrong predictions:", len(wrong))
print("Total correct predictions:", len(golden_df) - len(wrong))


# ==============================
# SHOW FIRST 20 WRONG CASES
# ==============================

print("\n========================================")
print("FIRST 20 WRONG PREDICTIONS")
print("========================================")

for i, row in wrong.head(20).iterrows():

    print("\n----------------------------------------")

    print("Customer message:")
    print(row["customer_message"])

    print("Actual intent:")
    print(row["actual_intent"])

    print("Predicted intent:")
    print(row["predicted_intent"])


# ==============================
# SAVE WRONG PREDICTIONS
# ==============================

wrong_file = "data/processed/golden_baseline_errors.csv"

wrong.to_csv(
    wrong_file,
    index=False
)

print("\n========================================")
print("DEBUG COMPLETED")
print("========================================")

print("Wrong predictions saved to:")
print(wrong_file)