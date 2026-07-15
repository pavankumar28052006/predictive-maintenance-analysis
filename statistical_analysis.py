import os
import pandas as pd
from scipy.stats import ttest_ind

import config

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

def run_hypothesis_tests(X, y):
    results = []
    print("\nStatistical Analysis")

    for column in SENSOR_COLUMNS:
        failed = X.loc[y == 1, column]
        normal = X.loc[y == 0, column]

        t_stat, p_value = ttest_ind(failed, normal, equal_var=False)
        significant = p_value < 0.05

        results.append({
            "feature": column,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": significant
        })

        print(f"\n{column}")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.6f}")

        if significant:
            print("Significant difference found")
        else:
            print("No significant difference found")

    results_df = pd.DataFrame(results)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    results_df.to_csv(
        os.path.join(config.OUTPUT_DIR, "hypothesis_test_results.csv"),
        index=False
    )
    return results_df