import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "maintenance_model.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TARGET = "Machine failure"

DROP_COLUMNS = [
    "UDI",
    "UID",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

RANDOM_STATE = 42

RF_PARAMS = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__class_weight": ["balanced"]
}