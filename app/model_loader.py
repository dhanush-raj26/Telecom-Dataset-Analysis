import joblib
import numpy as np

model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

def predict(data: list):
    arr = np.array(data).reshape(1, -1)
    arr = scaler.transform(arr)
    pred = model.predict(arr)
    prob = model.predict_proba(arr)[0][1]
    return int(pred[0]), float(prob)
