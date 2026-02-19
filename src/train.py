
"""
Training script (STARTER)
-------------------------
TODO:
1) Load data 
2) Fill NA in 'locality' and 'parking' with 'Missing'
3) Split X (features) and y (target='rent')
4) train_test_split
5) Build pipeline: preprocessor -> LinearRegression
6) Fit, evaluate (RMSE), and joblib.dump(model, 'model.pkl')
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import joblib

from .preprocess import build_preprocessor

def train(data_path = ?, model_path="model.pkl"):
    # TODO: Load CSV
    

    # TODO: Fill NA in categorical as 'Missing'
    
    # TODO: Split features/target
    
    # TODO: Train-test split
    
    # Build pipeline
    preprocessor = build_preprocessor()
    model_pipeline = ?

    # TODO: Fit model pipeline
    
    # TODO: Evaluate model
    
    # TODO: Save model

if __name__ == "__main__":
    train()
