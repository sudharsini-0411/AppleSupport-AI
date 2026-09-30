import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# -----------------------------------------------
# 1. Load training data and Golden Set
# -----------------------------------------------

train_df = pd.read_csv("data/processed/apple_support_balanced_v2.csv")
golden_df = pd.read_csv("data/processed/golden_evaluation_final.csv")

X_train = train_df["customer_message"].fillna("").tolist()
y_train = train_df["intent"]

X_test = golden_df["customer_message"].fillna("").tolist()
y_test = golden_df["corrected_intent"]   # trusted ground truth

print(f"Training samples : {len(X_train)}")
print(f"Golden Set size  : {len(X_test)}")

# -----------------------------------------------
# 2. Load Sentence Transformer
# -----------------------------------------------

print("\nLoading Sentence Transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

# -----------------------------------------------
# 3. Generate embeddings
# -----------------------------------------------

print("\nEncoding training messages...")
X_train_emb = embedder.encode(
    X_train,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nEncoding Golden Set messages...")
X_test_emb = embedder.encode(
    X_test,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"\nEmbedding shape: {X_train_emb.shape}")

# -----------------------------------------------
# 4. Train Logistic Regression
# -----------------------------------------------

print("\nTraining Logistic Regression...")
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_emb, y_train)

# -----------------------------------------------
# 5. Predict on Golden Set
# -----------------------------------------------

y_pred = clf.predict(X_test_emb)

# -----------------------------------------------
# 6. Metrics
# -----------------------------------------------

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1_w      = f1_score(y_test, y_pred, average="weighted", zero_division=0)
f1_macro  = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("\n=============================================")
print("  SENTENCE TRANSFORMER + LOGISTIC REGRESSION")
print("  Evaluated on 200-example Corrected Golden Set")
print("=============================================")
print(f"Accuracy          : {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall    (weighted): {recall:.4f}")
print(f"F1        (weighted): {f1_w:.4f}")
print(f"F1        (macro)   : {f1_macro:.4f}")

print("\n=============================================")
print("  CLASSIFICATION REPORT")
print("=============================================")
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------------------------
# 7. Save predictions for comparison and analysis
# -----------------------------------------------

results_df = golden_df[["tweet_id", "customer_message", "corrected_intent"]].copy()
results_df["st_predicted_intent"] = y_pred
results_df["correct"] = (results_df["corrected_intent"] == results_df["st_predicted_intent"])

results_df.to_csv(
    "data/processed/st_golden_predictions.csv",
    index=False
)

print("\nPredictions saved: data/processed/st_golden_predictions.csv")
print(f"Correct predictions : {results_df['correct'].sum()} / {len(results_df)}")
