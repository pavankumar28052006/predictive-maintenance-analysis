# Predictive Maintenance Analysis

This project is a machine learning based predictive maintenance system built using the AI4I 2020 Predictive Maintenance Dataset.

The main goal of the project is to analyze machine sensor data and predict whether a machine is likely to fail. Along with machine failure prediction, I also performed statistical analysis and K-Means clustering to better understand the machine operating conditions.

The dataset contains 10,000 machine records. Only 339 records represent machine failures, which is around 3.39% of the complete dataset. Since the dataset is highly imbalanced, I focused on metrics such as precision, recall, F1 score, and ROC-AUC instead of depending only on accuracy.

## Project Features

* Predicts machine failure using Random Forest
* Preprocesses numerical and categorical data
* Performs statistical hypothesis testing on sensor features
* Uses K-Means clustering to analyze machine operating states
* Selects the number of clusters using silhouette score
* Handles class imbalance using class weights
* Tunes the Random Forest model using GridSearchCV
* Evaluates the model using precision, recall, F1 score, and ROC-AUC
* Generates confusion matrix and feature importance plots
* Saves the trained model for predicting new machine data

## Project Structure

```text
predictive-maintenance-analysis/
│
├── data/
│   └── ai4i2020.csv
│
├── models/
│   └── maintenance_model.pkl
│
├── outputs/
│   ├── cluster_summary.csv
│   ├── confusion_matrix.png
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── hypothesis_test_results.csv
│   ├── machine_clusters.png
│   ├── model_metrics.csv
│   └── silhouette_scores.png
│
├── clustering.py
├── config.py
├── data_pipeline.py
├── predict.py
├── statistical_analysis.py
├── train.py
├── requirements.txt
└── README.md
```

## Dataset

I used the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

The dataset contains synthetic machine data that represents predictive maintenance scenarios in manufacturing industries.

### Dataset Details

* Total records: 10,000
* Target column: `Machine failure`
* Machine failure records: 339
* Failure rate: 3.39%
* Task: Binary Classification

The main input features used in this project are:

| Feature                 | Description                                  |
| ----------------------- | -------------------------------------------- |
| Type                    | Product quality type                         |
| Air temperature [K]     | Air temperature around the machine           |
| Process temperature [K] | Temperature during the manufacturing process |
| Rotational speed [rpm]  | Rotational speed of the machine              |
| Torque [Nm]             | Torque produced by the machine               |
| Tool wear [min]         | Tool usage time in minutes                   |

The original dataset also contains the following columns:

```text
UID
Product ID
TWF
HDF
PWF
OSF
RNF
```

`UID` and `Product ID` are identifier columns, so I removed them before training the model.

The columns `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` represent different machine failure types. Since the `Machine failure` target is based on these failure types, I removed these columns from the input features to avoid data leakage.

The original CSV file is not manually modified. These columns are removed in the data preprocessing code.

## Workflow

The overall workflow of the project is:

```text
Machine Dataset
       |
       v
Data Preprocessing
       |
       v
Train-Test Split
       |
       +----------------------+
       |                      |
       v                      v
Statistical Analysis     K-Means Clustering
       |                      |
       +----------------------+
                  |
                  v
           Random Forest
                  |
                  v
            GridSearchCV
                  |
                  v
          Model Evaluation
                  |
                  v
             Save Model
```

## Data Preprocessing

The preprocessing is implemented using Scikit-learn Pipeline and ColumnTransformer.

For numerical features, I used:

* Median imputation
* StandardScaler

For categorical features, I used:

* Most frequent value imputation
* One-hot encoding

The dataset does not currently contain missing values, but I included imputation in the pipeline so that the model can also handle missing values if similar machine data is used later.

I used a stratified 80-20 train-test split to maintain the same machine failure ratio in both training and testing data.

## Statistical Analysis

I performed Welch's t-test on the main machine sensor features.

The goal was to check whether the average sensor values are significantly different between failed and non-failed machines.

The features tested are:

* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear

For each feature, the hypotheses are:

```text
H0: The mean sensor value is the same for failed and non-failed machines.

H1: The mean sensor value is different for failed and non-failed machines.
```

I used a significance level of 0.05.

If the p-value is less than 0.05, the null hypothesis is rejected.

The results of the statistical analysis are saved in:

```text
outputs/hypothesis_test_results.csv
```

The hypothesis test is used to identify statistical differences between the two machine groups. It does not prove that a sensor feature directly causes machine failure.

## K-Means Clustering

I used K-Means clustering to explore different machine operating states based on sensor readings.

The following features are used for clustering:

* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear

Before applying K-Means, the features are standardized using StandardScaler.

Instead of directly selecting the number of clusters, I tested K values from 2 to 7.

For each value of K, I calculated the silhouette score. The K value with the highest silhouette score is selected for the final clustering model.

The clustering results are saved as:

```text
outputs/cluster_summary.csv
outputs/silhouette_scores.png
outputs/machine_clusters.png
```

The cluster summary shows the average sensor values for each machine operating cluster.

In this project, clustering is used only for data analysis. The cluster labels are not used as input features for the Random Forest model.

## Machine Failure Prediction

I used a Random Forest Classifier for machine failure prediction.

Random Forest was selected because it can handle nonlinear relationships between machine sensor features and also provides feature importance values.

The model uses class weight balancing because the number of machine failure records is much lower than the number of normal machine records.

The complete model pipeline includes:

```text
Data Preprocessing
        |
        v
Random Forest Classifier
```

## Hyperparameter Tuning

I used GridSearchCV to tune the Random Forest model.

The parameters tested include:

* Number of trees
* Maximum tree depth
* Minimum samples required to split a node

The model is evaluated using 5-fold cross-validation.

I used macro F1 score as the GridSearchCV scoring metric because the dataset is highly imbalanced.

Macro F1 gives equal importance to both the failure and non-failure classes.

## Model Evaluation

The final model is evaluated on the test dataset.

The following metrics are used:

### Precision

Precision shows how many machines predicted as failures actually failed.

### Recall

Recall shows how many actual machine failures were detected by the model.

For this project, recall is an important metric because missing an actual machine failure can result in unexpected equipment breakdown.

### F1 Score

F1 score provides a balance between precision and recall.

### ROC-AUC

ROC-AUC measures how well the model separates failed and non-failed machines.

### Confusion Matrix

The confusion matrix shows:

* True Negatives
* False Positives
* False Negatives
* True Positives

The evaluation results are saved in:

```text
outputs/model_metrics.csv
outputs/confusion_matrix.png
```

## Feature Importance

Random Forest feature importance is used to understand which machine features have more influence on the model predictions.

The complete feature importance values are stored in:

```text
outputs/feature_importance.csv
```

The top 10 features are visualized in:

```text
outputs/feature_importance.png
```

Feature importance is used only for model interpretation and does not mean that a feature directly causes machine failure.

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project folder:

```bash
cd predictive-maintenance-analysis
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Project

To train the model, run:

```bash
python train.py
```

The training script will:

1. Load the machine dataset
2. Remove identifier and failure type columns
3. Split the dataset into training and testing data
4. Perform statistical hypothesis testing
5. Run K-Means clustering
6. Preprocess the data
7. Tune the Random Forest model
8. Evaluate the model
9. Generate plots and CSV outputs
10. Save the trained model

The trained model is saved in:

```text
models/maintenance_model.pkl
```

## Making a Prediction

After training the model, run:

```bash
python predict.py
```

Example sensor input:

```python
sensor_data = {
    "Type": "L",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 200
}
```

The output displays the machine risk level and failure probability.

Example:

```text
Prediction Result
Status: LOW RISK
Failure Probability: 8.25%
```

The risk levels used in the project are:

```text
Failure Probability < 30%       -> LOW RISK
Failure Probability 30% - 70%   -> MODERATE RISK
Failure Probability >= 70%      -> HIGH RISK
```

These risk levels are used only for project demonstration. In a real industrial system, the thresholds should be selected based on machine maintenance cost, failure impact, and operational requirements.

## Generated Outputs

| File                          | Description                                    |
| ----------------------------- | ---------------------------------------------- |
| `model_metrics.csv`           | Model precision, recall, F1 score, and ROC-AUC |
| `confusion_matrix.png`        | Machine failure confusion matrix               |
| `feature_importance.csv`      | Random Forest feature importance values        |
| `feature_importance.png`      | Top machine failure predictors                 |
| `hypothesis_test_results.csv` | Statistical hypothesis test results            |
| `cluster_summary.csv`         | Average sensor values for each cluster         |
| `silhouette_scores.png`       | Silhouette score for different K values        |
| `machine_clusters.png`        | Machine operating state cluster visualization  |

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SciPy
* Matplotlib
* Joblib

## Future Improvements

Some improvements I would like to add in the future are:

* Compare Random Forest with Logistic Regression, SVM, and other classification models
* Optimize the classification threshold for better failure detection
* Use SHAP for better model explainability
* Apply anomaly detection for unusual machine conditions
* Build a dashboard for machine monitoring
* Deploy the model using FastAPI
* Test the workflow on real time-series sensor data

## Dataset Credit

This project uses the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository.

Dataset citation:

> AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository.

DOI: `10.24432/C5HS5C`

The dataset is available under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
