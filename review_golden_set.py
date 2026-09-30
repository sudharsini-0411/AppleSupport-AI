import pandas as pd

FILE = "data/processed/golden_evaluation_set.csv"

df = pd.read_csv(FILE)

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

for i in range(len(df)):

    print("\n" + "=" * 60)
    print(f"ROW {i + 1} / {len(df)}")
    print("=" * 60)

    print("\nCustomer message:")
    print(df.loc[i, "customer_message"])

    print("\nHistorical support response:")
    print(df.loc[i, "support_response"])

    print("\nSuggested intent:")
    print(df.loc[i, "suggested_intent"])

    print("\nChoose the correct intent:")

    for number, intent in enumerate(intents, start=1):
        print(f"{number}. {intent}")

    while True:
        choice = input("\nEnter intent number (1-11): ")

        if choice.isdigit() and 1 <= int(choice) <= 11:
            actual_intent = intents[int(choice) - 1]
            break

        print("Invalid choice. Enter a number from 1 to 11.")

    df.loc[i, "actual_intent"] = actual_intent

    print("\nSuggested decision:")
    print(df.loc[i, "suggested_decision"])

    print("\nChoose the correct decision:")
    print("1. AUTO-HANDLED")
    print("2. ESCALATED")

    while True:
        choice = input("\nEnter decision number (1-2): ")

        if choice == "1":
            decision = "AUTO-HANDLED"
            break
        elif choice == "2":
            decision = "ESCALATED"
            break

        print("Invalid choice. Enter 1 or 2.")

    df.loc[i, "expected_decision"] = decision

    # Use historical support response as expected response
    df.loc[i, "expected_response"] = df.loc[i, "support_response"]

    # Save after every row
    df.to_csv(FILE, index=False)

    print("\nSaved successfully!")

print("\n========================================")
print("GOLDEN SET REVIEW COMPLETED!")
print("========================================")
print("Total rows reviewed:", len(df))