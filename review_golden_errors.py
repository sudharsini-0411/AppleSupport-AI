import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ========================================
# FILES
# ========================================

train_file = "data/processed/apple_support_labeled.csv"
golden_file = "data/processed/golden_evaluation_set.csv"


# ========================================
# LOAD DATA
# ========================================

train_df = pd.read_csv(train_file)
golden_df = pd.read_csv(golden_file)

print("Training rows:", len(train_df))
print("Golden rows:", len(golden_df))


# ========================================
# TRAIN TF-IDF MODEL
# ========================================

X_train = train_df["customer_message"].fillna("")
y_train = train_df["intent"]

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)


# ========================================
# PREDICT GOLDEN SET
# ========================================

X_golden = golden_df["customer_message"].fillna("")

X_golden_tfidf = vectorizer.transform(X_golden)

golden_predictions = model.predict(X_golden_tfidf)

golden_df["predicted_intent"] = golden_predictions


# ========================================
# FIND DISAGREEMENTS
# ========================================

errors = golden_df[
    golden_df["actual_intent"] != golden_df["predicted_intent"]
].copy()

print("\n========================================")
print("GOLDEN SET LABEL REVIEW")
print("========================================")

print("Total Golden rows:", len(golden_df))
print("Rows needing review:", len(errors))


# ========================================
# VALID INTENTS
# ========================================

intents = [
    "battery_issue",
    "software_update",
    "app_issue",
    "account_issue",
    "payment_billing",
    "refund",
    "subscription",
    "connectivity_issue",
    "device_issue",
    "order_delivery",
    "other"
]


# ========================================
# REVIEW ERRORS
# ========================================

for number, (index, row) in enumerate(errors.iterrows(), start=1):

    print("\n" + "=" * 70)
    print(f"ERROR {number} / {len(errors)}")
    print("=" * 70)

    print("\nCustomer message:")
    print(row["customer_message"])

    print("\nHistorical support response:")
    print(row["support_response"])

    print("\nCurrent Golden intent:")
    print(row["actual_intent"])

    print("\nTF-IDF predicted intent:")
    print(row["predicted_intent"])

    print("\nChoose the CORRECT intent:")

    for i, intent in enumerate(intents, start=1):
        print(f"{i}. {intent}")

    while True:

        choice = input("\nEnter intent number (1-11), or 0 to keep current: ")

        if choice == "0":
            new_intent = row["actual_intent"]
            break

        if choice.isdigit() and 1 <= int(choice) <= 11:
            new_intent = intents[int(choice) - 1]
            break

        print("Invalid choice. Enter 1-11 or 0.")

    golden_df.loc[index, "actual_intent"] = new_intent

    golden_df.to_csv(
        golden_file,
        index=False
    )

    print("\n✅ Saved.")


# ========================================
# FINAL MESSAGE
# ========================================

print("\n========================================")
print("REVIEW COMPLETED")
print("========================================")

print("Golden rows:", len(golden_df))

print("\nUpdated file:")
print(golden_file)