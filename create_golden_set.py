import pandas as pd

# Input and output files
input_file = "data/processed/apple_support_clean.csv"
output_file = "data/processed/golden_evaluation_set.csv"

# Load dataset
df = pd.read_csv(input_file)

print("Total conversations:", len(df))

# Select 200 random conversations
golden = df.sample(
    n=200,
    random_state=42
).copy()

# Add columns for manual evaluation
golden["actual_intent"] = ""
golden["expected_response"] = ""
golden["expected_decision"] = ""

# Save
golden.to_csv(
    output_file,
    index=False
)

print("\n==============================")
print("Golden Evaluation Set Created!")
print("==============================")
print("Number of conversations:", len(golden))
print("Saved to:", output_file)
print("\nColumns:")
print(golden.columns.tolist())
