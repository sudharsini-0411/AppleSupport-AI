import pandas as pd

file = "data/processed/golden_review_working.csv"

df = pd.read_csv(file)
df["corrected_intent"] = df["corrected_intent"].fillna("").astype(str)

corrections = {
    41: "software_update",
    42: "software_update",
    43: "app_issue",
    44: "software_update",
    45: "other",
    46: "battery_issue",
    47: "battery_issue",
    48: "other",
    49: "device_issue",
    50: "software_update",
    51: "other",
    52: "app_issue",
    53: "other",
    54: "device_issue",
    55: "software_update",
    56: "app_issue",
    57: "battery_issue",
    58: "app_issue",
    59: "connectivity_issue",
}

for row, intent in corrections.items():
    df.loc[row - 1, "corrected_intent"] = intent

df.to_csv(file, index=False)

print("Corrections 41-59 saved")
print("Rows corrected:", len(corrections))
print()
print(df.iloc[40:59][
    ["customer_message", "actual_intent", "corrected_intent"]
].to_string(index=False))