# Telecom Churn Prediction — Technical Documentation

## 1. Overview
- **Purpose:** Predict whether a telecom customer will churn (leave service) using historical customer, service, and usage data. The model is intended to support retention strategies by identifying high-risk customers and explaining drivers of churn.

## 2. Problem Statement
- Given customer records (demographics, account details, usage, billing), build a classifier that outputs a churn probability for each customer. The output should be usable by business teams (probability + explanation + top drivers).

## 3. Dataset
- **Location:** [Telecom Churn Dataset.csv](Telecom%20Churn%20Dataset.csv)
- **Typical columns:** customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn
- **Notes:** Clean TotalCharges (some rows may have empty strings). Convert categorical fields to consistent categories. Handle missing values and outliers in numeric fields.

## 4. Project Structure
- **Repository root:** contains `churn_prediction_model.ipynb`, dataset, `Dockerfile`, `docker-compose.yaml`, and `requirements.txt`.
- **Training code:** [training/train.py](training/train.py) — orchestrates data loading, preprocessing, model training, evaluation, and persistence.
- **App:** [app/main.py](app/main.py) — REST API for serving predictions; [app/model_loader.py](app/model_loader.py) — loads serialized model(s); [app/models.py](app/models.py) & [app/schemas.py](app/schemas.py) — request/response data models and validation; [app/database.py](app/database.py) — optional persistence for requests and predictions.
- **Model storage:** trained model artifacts (e.g., `models/random_forest.joblib` or `model/`) — store versions with timestamps and metadata.

## 5. Why RandomForest (when chosen)
- **Simple working idea:** RandomForest is an ensemble of decision trees trained on random subsets of the data and/or features (bagging). Each tree votes; the forest averages votes or probabilities to form a prediction. Randomness reduces correlation between trees and the ensemble reduces variance.
- **Why use it:**
  - Robust to noisy features and outliers.
  - Handles numeric and categorical features (after encoding).
  - Provides feature importance scores for quick interpretability.
  - Good baseline with strong performance and few tuning needs.
  - Fast inference for realtime-ish APIs.

## 6. Simple RandomForest Explanation (non-technical)
- Imagine many simple rules (trees) built from small samples of customers. Each rule guesses whether a customer will churn. The RandomForest asks all rules and averages their answers — this reduces over-reliance on any single odd rule.

## 7. End-to-end Technical Workflow
1. Data ingestion
   - Load [Telecom Churn Dataset.csv](Telecom%20Churn%20Dataset.csv).
   - Basic validation (column presence, types, missing rows).
2. Preprocessing
   - Clean numeric fields (coerce `TotalCharges` to numeric, fill or drop NaNs).
   - Convert categorical columns: use one-hot or ordinal encoding; keep mapping saved to disk.
   - Feature creation: tenure buckets, avg charges per month, engagement metrics.
   - Scale numeric features if model requires it (RandomForest doesn't require scaling but consistent transforms are useful).
3. Train / Validation Split
   - Use stratified split on `Churn` to preserve class balance.
   - Consider time-based split if dataset has temporal ordering.
4. Model training
   - Baseline: `sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)`
   - Evaluate with cross-validation and/or holdout set.
   - Track hyperparameters: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
5. Evaluation
   - Metrics: ROC-AUC, Precision, Recall, F1, PR-AUC, Confusion Matrix, calibration (Brier score).
   - Business metrics: capture rate at top-k (e.g., how many churners are in top 5% predicted probability).
6. Explainability
   - Global: feature importances (from RandomForest), permutation importance.
   - Local: SHAP values or LIME for single-instance explanations.
7. Model persistence
   - Serialize model and preprocessing pipeline using `joblib` or `pickle` with versioned filenames and metadata (training date, metrics, hyperparameters).
   - Example path: `models/random_forest_v2026-02-16.joblib`.
8. Deployment
   - Serve using the API in [app/main.py](app/main.py) that uses [app/model_loader.py](app/model_loader.py) to load the model and preprocessing objects.
   - Containerize using `Dockerfile` and orchestrate via `docker-compose.yaml`.
9. Monitoring & Retraining
   - Log input distributions and prediction probabilities.
   - Monitor model performance over time (drift detection) and schedule retraining when performance degrades.

## 8. Example: RandomForest prediction for one instance
- **Input (JSON example):**

```json
{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 34,
  "PhoneService": "Yes",
  "InternetService": "Fiber optic",
  "Contract": "Month-to-month",
  "MonthlyCharges": 70.35,
  "TotalCharges": 2400.0
}
```

- **Processing steps:**
  - Apply same category encodings and numeric transforms used at training time.
  - Pass transformed vector to RandomForest's `predict_proba()`.
- **Sample output (explainable):**

```json
{
  "customerID": "7590-VHVEG",
  "churn_probability": 0.78,
  "predicted_churn": true,
  "top_drivers": [
    {"feature": "Contract_Month-to-month", "impact": "+0.20"},
    {"feature": "InternetService_Fiber optic", "impact": "+0.10"},
    {"feature": "MonthlyCharges", "impact": "+0.08"}
  ]
}
```

- **Interpretation:** probability 0.78 (78%) — likely to churn. `top_drivers` come from SHAP or local feature contribution method; positive impact numbers indicate increased churn probability attributable to that feature.

## 9. Reproducible commands
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Train model (example):

```bash
python training/train.py
```

- Run API locally:

```bash
python app/main.py
# or using Docker
docker-compose up --build
```

## 10. Deployment notes
- Expose a `/predict` endpoint that accepts validated JSON (use schemas in [app/schemas.py](app/schemas.py)).
- Include health and metrics endpoints (`/health`, `/metrics`).
- Use model versioning header or response field so consumers know which model served the prediction.

## 11. Monitoring and Maintenance
- Log features, predicted probability, and outcome when ground truth becomes available.
- Track population and feature drift; consider retraining triggers (time-based or performance-based).
- Maintain an artifacts registry with metrics and model metadata.

## 12. Testing & Validation
- Unit tests for preprocessing functions and schema validation.
- Integration test for API: send a sample request and check response shape and status code.

## 13. Security and Privacy
- Avoid logging PII in clear text. Use hashed IDs when persistence is needed.
- Apply access control to model-serving endpoints.

## 14. Next steps and extensions
- Try gradient boosting (XGBoost/LightGBM) for possible performance gains.
- Add explainability UI for business users (display SHAP waterfall per customer).
- Automate retraining pipeline and A/B testing of model versions.

---
Generated on 2026-02-16. For implementation pointers, see training code at [training/train.py](training/train.py) and serving code in [app/main.py](app/main.py).
