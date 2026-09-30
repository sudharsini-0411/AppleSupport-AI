import pandas as pd

df = pd.read_csv("data/processed/apple_support_clean.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 conversations:")
print(df[["customer_message", "support_response"]].head(10).to_string(index=False))