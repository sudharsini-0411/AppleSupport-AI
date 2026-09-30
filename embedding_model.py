import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

df = pd.read_csv(
    "data/processed/apple_support_labeled.csv"
)

print("Dataset size:", len(df))


X = df["customer_message"].fillna("")
y = df["intent"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))



print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


print("\nCreating training embeddings...")

X_train_embeddings = embedding_model.encode(
    X_train.tolist(),
    batch_size=32,
    show_progress_bar=True
)

print("\nCreating testing embeddings...")

X_test_embeddings = embedding_model.encode(
    X_test.tolist(),
    batch_size=32,
    show_progress_bar=True
)


print(
    "Embedding shape:",
    X_train_embeddings.shape
)

print("\nTraining classifier...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_embeddings,
    y_train
)

y_pred = model.predict(
    X_test_embeddings
)



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


print("\n================================")
print("Embedding + Logistic Regression")
print("================================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))



print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)