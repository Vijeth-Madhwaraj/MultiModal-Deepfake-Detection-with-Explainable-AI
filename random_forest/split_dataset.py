from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
PROJECT_DIR = Path(__file__).resolve().parent

METADATA_DIR = PROJECT_DIR / "metadata"

INPUT_CSV = METADATA_DIR / "dataset_metadata.csv"

TRAIN_CSV = METADATA_DIR / "train.csv"
VALID_CSV = METADATA_DIR / "validation.csv"
TEST_CSV = METADATA_DIR / "test.csv"
df = pd.read_csv(INPUT_CSV)

print(f"Total samples: {len(df)}")
print(df["label"].value_counts())
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42,
    shuffle=True
)
valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42,
    shuffle=True
)
train_df.to_csv(TRAIN_CSV, index=False)
valid_df.to_csv(VALID_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)
print("\nDataset Split Complete\n")

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(valid_df)}")
print(f"Test       : {len(test_df)}")

print("\nTrain Distribution")
print(train_df["label"].value_counts())

print("\nValidation Distribution")
print(valid_df["label"].value_counts())

print("\nTest Distribution")
print(test_df["label"].value_counts())