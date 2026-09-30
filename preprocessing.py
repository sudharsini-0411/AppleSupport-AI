import pandas as pd
import os

input_file = "data/raw/twcs.csv"
output_file = "data/processed/apple_support_conversations.csv"

chunksize = 100000

print("Step 1: Finding AppleSupport tweets...")

# --------------------------------------------------
# PASS 1: Get all AppleSupport tweet IDs
# --------------------------------------------------

apple_ids = set()

for chunk in pd.read_csv(input_file, chunksize=chunksize):

    apple = chunk[chunk["author_id"] == "AppleSupport"]

    apple_ids.update(apple["tweet_id"].astype(str))

print("AppleSupport tweet IDs:", len(apple_ids))


# --------------------------------------------------
# PASS 2: Find customer tweets that point to
# AppleSupport replies
# --------------------------------------------------

conversations = []

print("\nStep 2: Finding customer → AppleSupport conversations...")

for chunk in pd.read_csv(input_file, chunksize=chunksize):

    # Customer tweets
    customer = chunk[chunk["inbound"] == True].copy()

    # Customer tweet whose response is from AppleSupport
    customer["response_tweet_id"] = (
        customer["response_tweet_id"]
        .astype(str)
    )

    matched = customer[
        customer["response_tweet_id"].isin(apple_ids)
    ]

    if not matched.empty:

        conversations.append(
            matched[
                [
                    "tweet_id",
                    "text",
                    "response_tweet_id"
                ]
            ]
        )

    print("Processed:", len(chunk), "rows")


# --------------------------------------------------
# Combine customer messages
# --------------------------------------------------

if conversations:

    result = pd.concat(
        conversations,
        ignore_index=True
    )

    # --------------------------------------------------
    # Get AppleSupport response text
    # --------------------------------------------------

    print("\nStep 3: Getting AppleSupport responses...")

    apple_responses = []

    for chunk in pd.read_csv(input_file, chunksize=chunksize):

        apple = chunk[
            chunk["tweet_id"].astype(str).isin(apple_ids)
        ]

        if not apple.empty:
            apple_responses.append(
                apple[
                    [
                        "tweet_id",
                        "text"
                    ]
                ]
            )

    apple_responses = pd.concat(
        apple_responses,
        ignore_index=True
    )

    apple_responses = apple_responses.rename(
        columns={
            "tweet_id": "response_tweet_id",
            "text": "support_response"
        }
    )

    # Make IDs same type
    result["response_tweet_id"] = (
        result["response_tweet_id"].astype(str)
    )

    apple_responses["response_tweet_id"] = (
        apple_responses["response_tweet_id"].astype(str)
    )

    # --------------------------------------------------
    # Merge
    # --------------------------------------------------

    final = result.merge(
        apple_responses,
        on="response_tweet_id",
        how="inner"
    )

    # Rename customer message
    final = final.rename(
        columns={
            "text": "customer_message"
        }
    )

    # Keep only required columns
    final = final[
        [
            "tweet_id",
            "customer_message",
            "support_response"
        ]
    ]

    # Remove empty messages
    final = final.dropna(
        subset=[
            "customer_message",
            "support_response"
        ]
    )

    # Remove duplicates
    final = final.drop_duplicates()

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    final.to_csv(
        output_file,
        index=False
    )

    print("\n================================")
    print("DONE!")
    print("Conversation pairs:", len(final))
    print("Saved to:", output_file)
    print("================================")

else:

    print("No conversations found.")