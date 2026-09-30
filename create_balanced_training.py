import pandas as pd

input_file = "data/processed/apple_support_labeled.csv"
output_file = "data/processed/apple_support_balanced.csv"

df = pd.read_csv(input_file)

print("Original dataset:", len(df))

# Maximum number of samples for each intent
SAMPLES_PER_INTENT = 1500

balanced_parts = []

for intent in df["intent"].unique():

    intent_df = df[df["intent"] == intent]

    # Take up to 1500 samples
    if len(intent_df) > SAMPLES_PER_INTENT:
        intent_df = intent_df.sample(
            n=SAMPLES_PER_INTENT,
            random_state=42
        )

    balanced_parts.append(intent_df)

balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True
)

# Shuffle the final dataset
balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
balanced_df.to_csv(
    output_file,
    index=False
)

print("\n========================================")
print("BALANCED DATASET CREATED")
print("========================================")

print("Original rows:", len(df))
print("Balanced rows:", len(balanced_df))

print("\nIntent distribution:")
print(balanced_df["intent"].value_counts())

print("\nSaved to:")
print(output_file)