import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

def load_data():
    df = pd.read_csv(config.DATA_PATH)
    print("Dataset loaded")
    print("Shape:", df.shape)
    return df

def split_data(df):
    columns_to_drop = [
        col for col in config.DROP_COLUMNS
        if col in df.columns
    ]

    X = df.drop(columns=[config.TARGET] + columns_to_drop)
    y = df[config.TARGET]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

def get_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor