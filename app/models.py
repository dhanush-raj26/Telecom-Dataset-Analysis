from sqlalchemy import Column, Integer, Float
from .database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    churn_prediction = Column(Integer)
    probability = Column(Float)
