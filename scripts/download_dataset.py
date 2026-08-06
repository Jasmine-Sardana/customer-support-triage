from datasets import load_dataset

# Download dataset
dataset = load_dataset("Tobi-Bueck/customer-support-tickets")

print(dataset)

# Convert to pandas
df = dataset["train"].to_pandas()

print(df.head())

# Save locally
df.to_csv("customer_support_tickets.csv", index=False)

print("Dataset downloaded successfully!")
