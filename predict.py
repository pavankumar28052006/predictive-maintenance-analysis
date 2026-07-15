import joblib
import pandas as pd
import config

class MaintenancePredictor:
    def __init__(self):
        self.model = joblib.load(config.MODEL_PATH)

    def predict(self, sensor_data):
        data = pd.DataFrame([sensor_data])
        
        prediction = self.model.predict(data)[0]
        probabilities = self.model.predict_proba(data)[0]

        classes = list(self.model.classes_)
        failure_index = classes.index(1)
        failure_probability = probabilities[failure_index]

        if failure_probability >= 0.7:
            status = "HIGH RISK"
        elif failure_probability >= 0.3:
            status = "MODERATE RISK"
        else:
            status = "LOW RISK"

        return {
            "prediction": int(prediction),
            "status": status,
            "failure_probability": round(failure_probability, 4)
        }

if __name__ == "__main__":
    sensor_data = {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 200
    }

    predictor = MaintenancePredictor()
    result = predictor.predict(sensor_data)

    print("\nPrediction Result")
    print("Status:", result["status"])
    print("Failure Probability:", f"{result['failure_probability'] * 100:.2f}%")