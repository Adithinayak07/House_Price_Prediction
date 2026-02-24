# src/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from .database import create_table, get_connection
from fastapi import Query

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="House Price Prediction API")
create_table()

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

    # ---- Monitoring Layer ----
    model_version = "v1.0"  # Temporary (we'll link to MLflow later)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            area, rent, locality, BHK,
            facing, parking, bathrooms,
            prediction, model_version
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        features.area,
        features.rent,
        features.locality,
        features.BHK,
        features.facing,
        features.parking,
        features.bathrooms,
        float(prediction),
        model_version
    ))

    conn.commit()
    conn.close()

    return {
        "predicted_price_per_sqft": round(float(prediction), 2)
    }




@app.get("/history")
def get_history(limit: int = Query(10, description="Number of records to return")):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    columns = [
        "id", "area", "rent", "locality", "BHK",
        "facing", "parking", "bathrooms",
        "prediction", "model_version", "timestamp"
    ]

    results = [dict(zip(columns, row)) for row in rows]

    return {"history": results}


@app.get("/metrics")
def get_metrics():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(prediction) FROM predictions")
    avg_pred = cursor.fetchone()[0]

    cursor.execute("SELECT MAX(prediction) FROM predictions")
    max_pred = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(prediction) FROM predictions")
    min_pred = cursor.fetchone()[0]

    conn.close()

    return {
        "total_predictions": total,
        "average_prediction": avg_pred,
        "max_prediction": max_pred,
        "min_prediction": min_pred
    }