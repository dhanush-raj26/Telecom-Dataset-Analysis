import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("C:\study\PROJECTS\Telecom_Analysis\Telecom Churn Dataset.csv")

if "ID" in df.columns:
    df = df.drop(columns=["ID"])


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

binary_cols = [
    'Gender', 'Married', 'PhoneService', 'PaperlessBilling',
    'MultipleLines', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]

for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, columns=[
    'InternetService', 'Contract', 'PaymentMethod'
])

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

joblib.dump(model, "../model/churn_model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")

print("Model saved successfully.")
