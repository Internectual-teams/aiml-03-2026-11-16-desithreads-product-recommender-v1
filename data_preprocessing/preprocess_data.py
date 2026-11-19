
import pandas as pd
from pathlib import Path

# Define file paths
BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = BASE_DIR / "raw_products.csv"
CLEANED_FILE = BASE_DIR / "cleaned_products.csv"

# Read raw product data
df = pd.read_csv(RAW_FILE)

# Store original row count
rows_before = len(df)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Count duplicate product IDs
duplicates_before = df["product_id"].duplicated().sum()

# Count missing categorical values
categorical_columns = ["category", "color", "material"]
missing_before = df[categorical_columns].isna().sum().sum()

# Remove duplicate product IDs
df = df.drop_duplicates(subset="product_id", keep="first")

# Trim spaces from text columns
text_columns = ["product_id", "name", "category", "color", "material"]

for column in text_columns:
    df[column] = df[column].astype("string").str.strip()

# Replace empty strings with missing values
df[text_columns] = df[text_columns].replace("", pd.NA)

# Drop rows without product_id or name
df = df.dropna(subset=["product_id", "name"])

# Fill missing categorical values with Unknown
missing_filled = df[categorical_columns].isna().sum().sum()

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")

# Standardise casing
for column in categorical_columns:
    df[column] = df[column].str.title()

# Keep exactly five required columns
df = df[
    ["product_id", "name", "category", "color", "material"]
]

# Store final row count
rows_after = len(df)

# Save cleaned data
df.to_csv(CLEANED_FILE, index=False)

# Print cleaning summary
print("\n========== DATA CLEANING SUMMARY ==========")
print(f"Rows before cleaning: {rows_before}")
print(f"Rows after cleaning: {rows_after}")
print(f"Duplicate product IDs removed: {duplicates_before}")
print(f"Missing categorical values filled: {missing_filled}")
print(f"Rows removed: {rows_before - rows_after}")
print(f"Final columns: {list(df.columns)}")
print("\nCleaning completed successfully!")
print(f"Cleaned file saved at: {CLEANED_FILE}")
