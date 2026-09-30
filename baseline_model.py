import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ---------------------------------------
# 1. Load labeled dataset
# ---------------------------------------

df = pd.read_csv(
    "data/processed/apple_support_labeled.csv"
)

print("Dataset size:", len(df))


# ---------------------------------------
# 2. Input and target
# ---------------------------------------

X = df["customer_message"]
y = df["intent"]


# ---------------------------------------
# 3. Train/Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------
# 4. TF-IDF
# ---------------------------------------

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF shape:", X_train_tfidf.shape)


# ---------------------------------------
# 5. Logistic Regression
# ---------------------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)


# ---------------------------------------
# 6. Prediction
# ---------------------------------------

y_pred = model.predict(X_test_tfidf)


# ---------------------------------------
# 7. Evaluation
# ---------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n==============================")
print("TF-IDF + Logistic Regression")
print("==============================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# ---------------------------------------
# 8. Detailed report
# ---------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)