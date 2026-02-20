# src/train.py

import pandas as pd
import argparse
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor
from src.evaluate import evaluate_model


def train(data_path, model_path="model.pkl"):

    df = pd.read_csv(data_path)

    y = df["price_per_sqft"]
    X = df.drop("price_per_sqft", axis=1)

    test_size = 0.1
    random_state = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    preprocessor = build_preprocessor(df)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=10000),
    }

    best_rmse = float("inf")
    best_model_name = None
    best_pipeline = None

    print("\nModel Performance:\n")

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            pipeline = Pipeline(
                steps=[
                    ("Preprocessor", preprocessor),
                    ("Regressor", model)
                ]
            )

            pipeline.fit(X_train, y_train)

            rmse, r2 = evaluate_model(pipeline, X_test, y_test)

            print(f"{name} RMSE: {rmse:.2f}")
            print(f"{name} R2: {r2:.4f}\n")

            # 🔹 Log parameters
            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)

            # 🔹 Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            # 🔹 Log model artifact
            mlflow.sklearn.log_model(pipeline, "model")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                best_pipeline = pipeline

    print("===================================")
    print(f"Best Model: {best_model_name}")
    print(f"Best RMSE: {best_rmse:.2f}")
    print("===================================")

    joblib.dump(best_pipeline, model_path)

    print(f"\nSaved Model: {best_model_name}")
    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="model.pkl")

    args = parser.parse_args()

    train(args.data, args.model)