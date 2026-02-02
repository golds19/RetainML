"""
FastAPI prediction service for churn prediction.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from serving.predict import ChurnPredictor

predictor: Optional[ChurnPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = ChurnPredictor(
        models_dir=str(project_root / "models"),
        config_dir=str(project_root / "config"),
    )
    yield


app = FastAPI(title="RetainML Churn Prediction API", lifespan=lifespan)


class CustomerInput(BaseModel):
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: str = "840.0"
    gender: str = "Male"


class PredictionOutput(BaseModel):
    prediction: int
    churn_probability: float
    risk_level: str
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = predictor.predict(customer.model_dump())
    return result
