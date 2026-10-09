from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "hamrobazaar_extracted_products.csv"
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "kinbech"
    / "models"
    / "random_forest_model.pkl"
)
CATEGORICAL_FEATURES = ["Category", "Condition"]
NUMERIC_FEATURES = ["Storage_GB", "Age_years"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def main():
    listings = pd.read_csv(DATA_PATH)
    required_columns = set(FEATURES + ["Price_NPR"])
    missing_columns = required_columns.difference(listings.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing_columns))}")

    training_data = listings[FEATURES + ["Price_NPR"]].dropna()
    if training_data.empty:
        raise ValueError("Dataset has no complete rows for model training")

    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numbers", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(training_data[FEATURES], training_data["Price_NPR"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Trained on {len(training_data)} listings")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
