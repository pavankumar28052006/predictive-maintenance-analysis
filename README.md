# Predictive Maintenance Analysis

Predictive Maintenance Analysis is a compact machine learning project built around the AI4I 2020 predictive maintenance dataset. It combines preprocessing, statistical analysis, clustering, and supervised learning to estimate machine failure risk from sensor readings and operating conditions.

The dataset used in this project contains 10,000 rows and 14 columns, with a highly imbalanced target: 339 machine failures (3.39%). The training pipeline is designed with that imbalance in mind and reports metrics that are useful for failure detection, not just overall accuracy.

## What This Project Does

- Trains a Random Forest classifier to predict whether a machine will fail.
- Preprocesses numeric and categorical features with imputation, scaling, and one-hot encoding.
- Runs a small grid search to tune the classifier.
- Performs hypothesis tests on the main sensor features.
- Uses K-Means clustering to explore operating states.
- Saves evaluation outputs and visualizations to the `outputs/` folder.

## Project Structure

- `data/ai4i2020.csv` - source dataset.
- `config.py` - shared paths, target column, and model parameters.
- `data_pipeline.py` - data loading, train/test split, and preprocessing.
- `train.py` - end-to-end training, evaluation, clustering, and artifact saving.
- `predict.py` - simple wrapper for loading the trained model and scoring a new machine record.
- `statistical_analysis.py` - Welch t-tests for key sensor features.
- `clustering.py` - K-Means clustering and cluster summaries.
- `models/` - saved model files.
- `outputs/` - metrics, plots, and analysis artifacts.

## Requirements

- Python 3.9 or newer is recommended.
- Packages are listed in `requirements.txt`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How To Run

### 1. Train the model

Run the main training script from the project root:

```bash
python train.py
```

This will:

- load `data/ai4i2020.csv`,
- split the data into training and testing sets,
- run hypothesis tests and clustering on the training data,
- fit and tune a Random Forest model,
- evaluate the model on the test set,
- save the trained model to `models/maintenance_model.pkl`,
- write plots and CSV summaries into `outputs/`.

### 2. Make a prediction

After training, you can run the example prediction script:

```bash
python predict.py
```

The script loads the saved model and predicts failure risk for a sample machine record. You can also adapt the `sensor_data` dictionary in `predict.py` to score your own input.

## Generated Outputs

The training pipeline writes the following artifacts to `outputs/`:

- `model_metrics.csv` - precision, recall, F1, and ROC-AUC.
- `confusion_matrix.png` - confusion matrix for the test set.
- `feature_importance.csv` - feature importance values from the trained model.
- `feature_importance.png` - bar chart of the top predictors.
- `hypothesis_test_results.csv` - Welch t-test results for sensor features.
- `cluster_summary.csv` - per-cluster feature averages.
- `silhouette_scores.png` - silhouette scores across cluster counts.
- `machine_clusters.png` - cluster visualization for rotational speed vs torque.

## Modeling Approach

The training pipeline in `train.py` uses:

- median imputation and standard scaling for numeric features,
- most-frequent imputation and one-hot encoding for categorical features,
- a `RandomForestClassifier`,
- `GridSearchCV` with macro F1 scoring,
- a fixed random seed for reproducibility.

This makes the project easy to reproduce while still keeping the code compact enough to inspect and modify.

## Notes

- The dataset includes both general sensor readings and failure-type flags.
- The main target is `Machine failure`.
- The project keeps generated models and analysis artifacts out of the source code, so rerunning training is the intended way to refresh results.

## License

No license file is included in this repository. Add one if you plan to publish or share the project publicly.
