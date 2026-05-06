from pydantic import BaseModel

class CustomerInput(BaseModel):
    Gender: int
    Married: int
    Tenure: float
    MonthlyCharges: float
    TotalCharges: float
