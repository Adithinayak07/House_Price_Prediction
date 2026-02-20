# src/preprocess.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def build_preprocessor(df):

    
    continuous_features = ["area", "rent"]
    categorical_features = ["locality", "facing", "parking"]
    ordinal_features = ["BHK", "bathrooms"]

    # EXACT notebook logic
    continuous_features = (
        df.drop(columns=ordinal_features)
          .drop(columns=categorical_features)
          .columns.tolist()
    )

    if 'price_per_sqft' in continuous_features:
        continuous_features.remove('price_per_sqft')

    # Pipelines
    ordinal_transformer = Pipeline(
        steps=[("ordinalenc", OrdinalEncoder())]
    )

    categorical_transformer = Pipeline(
        steps=[("onehotenc", OneHotEncoder(handle_unknown="ignore"))]
    )

    continuous_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_transformer, continuous_features),
            ("categorical", categorical_transformer, categorical_features),
            ("ordinal", ordinal_transformer, ordinal_features),
        ],
        remainder="passthrough"
    )

    return preprocessor