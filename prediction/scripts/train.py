import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import json

# Load config
with open('../config.json', 'r') as file:
    config = json.load(file)

# Load dataset
df = pd.read_csv(config["server"]["prediction"]["wildfire_dataset_path"])

# -----------------------
# FEATURES (UPDATED TO MATCH DATASET)
# -----------------------
FEATURES = [
    "latitude",
    "longitude",
    "temp",
    "humidity",
    "wind",
    "rain",
    "day_of_year",
    "soil_moisture"  # <-- Changed from vegetation_index to match your real data column
]

# Drop rows with missing values in features or target
df = df.dropna(subset=FEATURES + ["fire"])

# -----------------------
# DIAGNOSTIC: Check Class Distribution
# -----------------------
class_counts = df["fire"].value_counts()
unique_classes = df["fire"].nunique()

print("--- Dataset Diagnostics ---")
print("Class distribution in 'fire' column:")
print(class_counts)
print("---------------------------\n")

if unique_classes < 2:
    print("WARNING: Your dataset only contains one class.")
    print("A classifier requires both '0' (no fire) and '1' (fire) classes to learn patterns.")
    print("Please check your training data and ensure negative cases are included.\n")

X = df[FEATURES]
y = df["fire"]

# -----------------------
# SPLIT
# -----------------------
# Stratify only if we have at least 2 classes and enough samples
if unique_classes >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

# -----------------------
# MODEL
# -----------------------
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced" if unique_classes >= 2 else None,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Evaluate
if unique_classes >= 2:
    y_pred = model.predict(X_test)
    print("Evaluation Metrics:")
    print(classification_report(y_test, y_pred))
else:
    print("Skipping classification report (only 1 class present in training data).")

# Save
joblib.dump(model, config["server"]["prediction"]["model_path"])
print("Model saved.")