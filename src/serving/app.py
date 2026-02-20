"""
FastAPI prediction service for churn prediction.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from serving.predict import ChurnPredictor
from monitoring.prediction_logger import PredictionLogger

predictor: Optional[ChurnPredictor] = None
prediction_logger: Optional[PredictionLogger] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, prediction_logger
    predictor = ChurnPredictor(
        models_dir=str(project_root / "models"),
        config_dir=str(project_root / "config"),
    )
    data_dir = Path(os.environ.get("DATA_DIR", str(project_root / "data")))
    prediction_logger = PredictionLogger(data_dir=data_dir)
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


class Concern(BaseModel):
    feature: str
    description: str
    impact_score: float


class RetentionAction(BaseModel):
    action: str
    reason: str
    priority: str


class PredictionOutput(BaseModel):
    prediction: int
    churn_probability: float
    risk_level: str
    model: str
    concerns: List[Concern] = []
    retention_plan: List[RetentionAction] = []


class PredictionRecord(BaseModel):
    id: int
    timestamp: str
    inputs: dict
    prediction: int
    churn_probability: float
    risk_level: str
    model_name: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = predictor.predict(customer.model_dump())
    if prediction_logger is not None:
        prediction_logger.log(
            inputs=customer.model_dump(),
            prediction=result["prediction"],
            churn_probability=result["churn_probability"],
            risk_level=result["risk_level"],
            model_name=result["model"],
        )
    return result


@app.get("/history", response_model=List[PredictionRecord])
def history(n: int = 50):
    if prediction_logger is None:
        raise HTTPException(status_code=503, detail="Logger not initialized")
    return prediction_logger.get_recent(n=n)
