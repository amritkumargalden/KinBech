"""Train and version the KinBech price-estimation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .data_preprocessing import RAW_DATA_PATH, PROCESSED_DATA_PATH, preprocess_csv
except ImportError:
    from data_preprocessing import RAW_DATA_PATH, PROCESSED_DATA_PATH, preprocess_csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "kinbech" / "models" / "random_forest_model.pkl"
METADATA_PATH = PROJECT_ROOT / "kinbech" / "models" / "model_metadata.json"
CATEGORICAL_FEATURES = ("Category", "Condition", "Brand")
NUMERIC_FEATURES = ("Storage_GB", "Age_years")
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Price_NPR"


def _atomic_write(path: Path, writer: Callable[[BinaryIO], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer(temporary)
        if temporary_path is None:
            raise RuntimeError("Temporary artifact file was not created")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def train_model(
    source: Path = RAW_DATA_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> dict:
    """Preprocess source listings, train, and persist a versioned full pipeline."""
    listings = preprocess_csv(source, processed_path)
    training_data = listings.loc[:, [*FEATURES, TARGET]].copy()
    for column in NUMERIC_FEATURES:
        training_data[column] = pd.to_numeric(training_data[column], errors="coerce")
    training_data[TARGET] = pd.to_numeric(training_data[TARGET], errors="coerce")
    training_data = training_data.dropna(subset=[*FEATURES, TARGET])
    training_data = training_data[training_data[TARGET] > 0]
    if training_data.empty:
        raise ValueError(f"No complete, positive-price listings found in {source}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
            ("numbers", StandardScaler(), list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
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
    model.fit(training_data[list(FEATURES)], training_data[TARGET])

    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()
    version_seed = f"{source_digest}:{timestamp}:{sklearn.__version__}"
    metadata = {
        "model_version": hashlib.sha256(version_seed.encode("utf-8")).hexdigest()[:16],
        "trained_at": timestamp,
        "training_rows": int(len(training_data)),
        "target": TARGET,
        "feature_columns": list(FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "preprocessing_steps": {
            "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
            "numeric_scaling": "StandardScaler",
        },
        "source_sha256": source_digest,
        "scikit_learn_version": sklearn.__version__,
    }

    _atomic_write(model_path, lambda file: pickle.dump(model, file))
    _atomic_write(
        metadata_path,
        lambda file: file.write(json.dumps(metadata, indent=2).encode("utf-8")),
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--processed-output", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata-output", type=Path, default=METADATA_PATH)
    args = parser.parse_args()
    metadata = train_model(
        source=args.source,
        processed_path=args.processed_output,
        model_path=args.model_output,
        metadata_path=args.metadata_output,
    )
    print(
        f"Trained model {metadata['model_version']} on "
        f"{metadata['training_rows']} listings"
    )
    print(f"Saved pipeline to {args.model_output}")
    print(f"Saved metadata to {args.metadata_output}")


if __name__ == "__main__":
    main()
