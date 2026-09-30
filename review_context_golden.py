import pandas as pd

input_file = "data/processed/golden_evaluation_context.csv"
output_file = "data/processed/golden_evaluation_context_reviewed.csv"

df = pd.read_csv(input_file)

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

print("Total rows:", len(df))

for i in range(len(df)):

    print("\n" + "=" * 80)
    print(f"ROW {i + 1} / {len(df)}")
    print("=" * 80)

    print("\nCONVERSATION:")
    print(df.iloc[i]["conversation_context"])

    print("\nCURRENT CUSTOMER MESSAGE:")
    print(df.iloc[i]["customer_message"])

    print("\nSUPPORT RESPONSE:")
    print(df.iloc[i]["support_response"])

    print("\nCURRENT INTENT:")
    print(df.iloc[i]["actual_intent"])

    print("\nChoose the correct intent:")
    for number, intent in enumerate(intents, start=1):
        print(f"{number}. {intent}")

    while True:
        choice = input(
            "\nEnter number (1-11), "
            "or press Enter to keep current: "
        ).strip()

        if choice == "":
            break

        if choice.isdigit() and 1 <= int(choice) <= 11:
            df.at[i, "actual_intent"] = intents[int(choice) - 1]
            break

        print("Invalid choice. Enter a number from 1 to 11.")

    df.to_csv(output_file, index=False)

print("\n===================================")
print("REVIEW COMPLETED")
print("===================================")
print("Rows reviewed:", len(df))
print("Saved to:")
print(output_file)