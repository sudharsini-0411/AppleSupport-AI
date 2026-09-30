import pandas as pd

input_file = "data/raw/twcs.csv"
output_file = "data/processed/apple_support_context.csv"

print("Loading dataset...")

df = pd.read_csv(input_file)

print("Total tweets:", len(df))

# Create lookup
tweet_lookup = df.set_index("tweet_id").to_dict("index")

# AppleSupport tweets
apple_support = df[df["author_id"] == "AppleSupport"]

print("AppleSupport tweets:", len(apple_support))

results = []


def get_context(tweet_id, max_turns=6):

    context = []
    current_id = tweet_id

    for _ in range(max_turns):

        if current_id not in tweet_lookup:
            break

        row = tweet_lookup[current_id]

        text = str(row["text"])

        if row["inbound"] == True:
            speaker = "Customer"
        else:
            speaker = "AppleSupport"

        context.append(f"{speaker}: {text}")

        parent_id = row["in_response_to_tweet_id"]

        if pd.isna(parent_id):
            break

        current_id = int(parent_id)

    context.reverse()

    return "\n".join(context)


print("Building conversation context...")

for _, support_row in apple_support.iterrows():

    support_id = int(support_row["tweet_id"])

    parent_id = support_row["in_response_to_tweet_id"]

    if pd.isna(parent_id):
        continue

    customer_id = int(parent_id)

    if customer_id not in tweet_lookup:
        continue

    customer_row = tweet_lookup[customer_id]

    # Make sure parent is a customer tweet
    if customer_row["inbound"] != True:
        continue

    customer_message = str(customer_row["text"])
    support_response = str(support_row["text"])

    context = get_context(customer_id, max_turns=6)

    results.append({
        "tweet_id": customer_id,
        "customer_message": customer_message,
        "support_response": support_response,
        "conversation_context": context
    })


result_df = pd.DataFrame(results)

result_df = result_df.drop_duplicates(
    subset=["tweet_id"]
)

result_df.to_csv(
    output_file,
    index=False
)

print("\n===================================")
print("APPLE SUPPORT CONTEXT CREATED")
print("===================================")

print("Conversation pairs:", len(result_df))

print("Columns:")
print(result_df.columns.tolist())

print("\nSaved to:")
print(output_file)