import pandas as pd

golden_file = "data/processed/golden_evaluation_set.csv"
context_file = "data/processed/apple_support_context.csv"
output_file = "data/processed/golden_evaluation_context.csv"

golden = pd.read_csv(golden_file)
context = pd.read_csv(context_file)

print("Golden rows:", len(golden))
print("Context rows:", len(context))

# Match using tweet_id
context_small = context[
    ["tweet_id", "conversation_context"]
].drop_duplicates(
    subset=["tweet_id"]
)

golden_context = golden.merge(
    context_small,
    on="tweet_id",
    how="left"
)

missing = golden_context["conversation_context"].isna().sum()

print("\n===================================")
print("CONTEXT MATCHING")
print("===================================")
print("Golden rows:", len(golden_context))
print("Context found:", len(golden_context) - missing)
print("Context missing:", missing)

golden_context.to_csv(
    output_file,
    index=False
)

print("\nSaved to:")
print(output_file)