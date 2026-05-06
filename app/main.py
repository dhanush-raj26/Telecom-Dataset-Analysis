from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from . import models, database
from .schemas import CustomerInput
from .model_loader import predict

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Telecom Churn Prediction API")

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "Churn API Running"}

@app.post("/predict")
def predict_churn(data: CustomerInput, db: Session = Depends(get_db)):
    
    input_data = [
        data.Gender,
        data.Married,
        data.Tenure,
        data.MonthlyCharges,
        data.TotalCharges
    ]

    prediction, probability = predict(input_data)

    db_record = models.Prediction(
        churn_prediction=prediction,
        probability=probability
    )

    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return {
        "prediction": prediction,
        "probability": probability,
        "id": db_record.id
    }
