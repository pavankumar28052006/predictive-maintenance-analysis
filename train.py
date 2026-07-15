import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import config
from clustering import run_clustering
from data_pipeline import get_preprocessor, load_data, split_data
from statistical_analysis import run_hypothesis_tests

def train():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("\nTraining class distribution")
    print(y_train.value_counts())

    run_hypothesis_tests(X_train, y_train)
    run_clustering(X_train)

    preprocessor = get_preprocessor(X_train)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=config.RANDOM_STATE))
    ])

    search = GridSearchCV(
        pipeline,
        config.RF_PARAMS,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    print("\nTraining model...")
    search.fit(X_train, y_train)
    
    model = search.best_estimator_

    print("\nBest parameters")
    print(search.best_params_)
    print(f"Best cross-validation F1: {search.best_score_:.4f}")

    evaluate_model(model, X_test, y_test)
    save_feature_importance(model)
    joblib.dump(model, config.MODEL_PATH)
    print("\nModel saved to:", config.MODEL_PATH)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\nModel Evaluation\n")
    print(classification_report(y_test, predictions))
    print(f"Failure Precision: {precision:.4f}")
    print(f"Failure Recall: {recall:.4f}")
    print(f"Failure F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    metrics = pd.DataFrame({
        "metric": ["precision", "recall", "f1_score", "roc_auc"],
        "score": [precision, recall, f1, roc_auc]
    })
    metrics.to_csv(os.path.join(config.OUTPUT_DIR, "model_metrics.csv"), index=False)

    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title("Machine Failure Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

def save_feature_importance(model):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier.feature_importances_
    })
    
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(os.path.join(config.OUTPUT_DIR, "feature_importance.csv"), index=False)

    top_features = importance_df.head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Importance")
    plt.title("Top Machine Failure Predictors")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "feature_importance.png"))
    plt.close()

if __name__ == "__main__":
    train()