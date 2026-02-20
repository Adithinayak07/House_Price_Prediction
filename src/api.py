# src/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="House Price Prediction API")


# Define input schema
class HouseFeatures(BaseModel):
    area: float
    rent: float
    locality: str
    BHK: int
    facing: str
    parking: str
    bathrooms: int


@app.get("/")
def root():
    return {"message": "House Price Prediction API is running 🚀"}


@app.post("/predict")
def predict(features: HouseFeatures):
    input_df = pd.DataFrame([features.dict()])

    prediction = model.predict(input_df)[0]

    return {
        "predicted_price_per_sqft": round(float(prediction), 2)
    }