"""Clean and enrich the listing dataset used by KinBech."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "Data" / "hamrobazaar_extracted_products.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "Data" / "data" / "processed_dataset.csv"

REQUIRED_COLUMNS = (
    "Category",
    "Title",
    "Price_NPR",
    "Condition",
    "Brand",
    "Model",
    "Storage_GB",
    "Age_years",
    "Location",
)
CONDITION_SCORES = {
    "new": 10.0,
    "like new": 9.0,
    "excellent": 8.0,
    "good": 6.0,
    "used": 5.0,
    "fair": 3.0,
}
NUMERIC_COLUMNS = ("Price_NPR", "Storage_GB", "Age_years")


def _parse_numeric(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    values = values.str.replace(",", "", regex=False)
    values = values.str.replace(r"[^\d.eE+-]", "", regex=True)
    return pd.to_numeric(values, errors="coerce")


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()


def preprocess_listings(listings: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, deterministic listing frame without mutating the input."""
    missing = sorted(set(REQUIRED_COLUMNS).difference(listings.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    cleaned = listings.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in ("Category", "Title", "Condition", "Brand", "Model", "Location"):
        cleaned[column] = _clean_text(cleaned[column])
        cleaned[column] = cleaned[column].replace("", pd.NA)
    for column in NUMERIC_COLUMNS:
        cleaned[column] = _parse_numeric(cleaned[column])

    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    cleaned = cleaned[
        (cleaned["Price_NPR"] > 0)
        & (cleaned["Storage_GB"] >= 0)
        & (cleaned["Age_years"] >= 0)
        & (cleaned["Age_years"] <= 50)
    ]
    cleaned = cleaned.drop_duplicates(keep="first")

    cleaned["Condition_Score"] = (
        cleaned["Condition"].str.casefold().map(CONDITION_SCORES).astype("Float64")
    )
    cleaned["Age_months"] = (cleaned["Age_years"] * 12).round().astype("Int64")
    cleaned["Price_per_GB"] = (
        cleaned["Price_NPR"] / cleaned["Storage_GB"].where(cleaned["Storage_GB"] > 0)
    ).round(2)
    return cleaned.reset_index(drop=True)


def preprocess_csv(
    source: Path = RAW_DATA_PATH,
    destination: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Process a CSV and atomically replace the processed dataset."""
    listings = pd.read_csv(source)
    cleaned = preprocess_listings(listings)
    if cleaned.empty:
        raise ValueError(f"No valid listings found in {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        cleaned.to_csv(temporary_path, index=False)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_PATH)
    args = parser.parse_args()
    cleaned = preprocess_csv(args.source, args.output)
    print(f"Processed {len(cleaned)} listings into {args.output}")


if __name__ == "__main__":
    main()
