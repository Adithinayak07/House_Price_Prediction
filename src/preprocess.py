from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np


def build_preprocessor():

    categorical_features = ["locality", "facing", "parking"]
    numerical_features = ["area", "BHK", "bathrooms"]

    # Log transform only for area
    area_log_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log_transform", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler())
    ])

    other_numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("area", area_log_pipeline, ["area"]),
        ("num", other_numeric_pipeline, ["BHK", "bathrooms"]),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor