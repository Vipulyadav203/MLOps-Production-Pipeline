import logging
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection Inference API")

class Transaction(BaseModel):
    amount: float
    time: float
    v1: float
    v2: float
    # Add other feature columns as needed

@app.on_event("startup")
def load_model():
    """
    Loads the trained model on startup.
    In production, this would load from a model registry like MLflow.
    """
    try:
        # For demonstration, we assume model.joblib exists.
        # global model
        # model = joblib.load("model.joblib")
        logger.info("Inference API starting up.")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

@app.post("/predict")
async def predict(transaction: Transaction):
    """
    Predicts if a transaction is fraudulent.
    """
    logger.info(f"Received prediction request: {transaction}")
    try:
        # Convert input to model-ready format
        data = [transaction.dict().values()]
        
        # Simulated prediction
        # prediction = model.predict(data)
        prediction = 0 # Placeholder
        
        return {"prediction": int(prediction), "status": "success"}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}
