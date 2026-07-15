# Predictive Maintenance & Machine Failure Analysis

This project implements an end-to-end Machine Learning pipeline to predict manufacturing machine failures based on IoT telemetry data.

## Key Features
- **Statistical Analysis**: Performed Welch's t-tests on sensor telemetry to validate feature significance.
- **Advanced Preprocessing**: Used `ColumnTransformer` to manage heterogeneous data (numeric sensor readings + categorical machine types).
- **Unsupervised Learning**: Implemented a custom K-Means transformer to segment machine operating states (optimized via silhouette scoring).
- **Predictive Modeling**: Trained a class-balanced Random Forest classifier using 5-fold GridSearchCV (Macro F1 scoring).
- **Deployment-Ready**: Includes a production-ready inference module (`predict.py`) and a modularized pipeline architecture.

## Evaluation Metrics (Test Set)
- **ROC-AUC**: 0.9707
- **Failure F1 Score**: 0.6569
- **Approach**: Class-weight balancing to handle significant data imbalance.
