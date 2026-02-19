import pandas as pd
import argparse
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor


def train(data_path, model_path="model.pkl"):

    # Load data
    df = pd.read_csv(data_path)

    # Separate target and features
    y = df["rent"]
    X = df.drop(columns=["rent"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build preprocessor
    preprocessor = build_preprocessor()

    # Models to compare
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),
    }

    best_rmse = float("inf")
    best_model_name = None
    best_pipeline = None

    print("\nModel Performance:\n")

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # 🔥 Log-transform target
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)

        # Train on log target
        pipeline.fit(X_train, y_train_log)

        # Predict (log scale)
        preds_log = pipeline.predict(X_test)

        # Convert back to original scale
        preds = np.expm1(preds_log)

        # Evaluate on original scale
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds)

        print(f"{name} RMSE: {rmse:.2f}")
        print(f"{name} R2: {r2:.4f}\n")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_pipeline = pipeline

    print("===================================")
    print(f"Best Model: {best_model_name}")
    print(f"Best RMSE: {best_rmse:.2f}")
    print("===================================")

    # Save best model pipeline
    joblib.dump(best_pipeline, model_path)

    print(f"\nSaved Model: {best_model_name}")
    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--model", default="model.pkl", help="Model save path")

    args = parser.parse_args()

    train(args.data, args.model)