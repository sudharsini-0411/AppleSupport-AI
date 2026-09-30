import pandas as pd

file = "data/processed/golden_evaluation_set.csv"

df = pd.read_csv(file)

# Copy automatic suggestions into the Golden Set
df["actual_intent"] = df["suggested_intent"]
df["expected_response"] = df["suggested_response"]
df["expected_decision"] = df["suggested_decision"]

# Save the updated file
df.to_csv(file, index=False)

print("Golden Set pre-filled successfully!")
print("Rows:", len(df))