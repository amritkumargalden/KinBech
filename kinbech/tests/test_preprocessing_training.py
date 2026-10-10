import json
import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from kinbech.data_preprocessing import preprocess_listings
from kinbech.model_training import train_model


def listing(title, price, brand="TestBrand", condition="Excellent", age=1):
    return {
        "Category": "Mobile",
        "Title": title,
        "Price_NPR": price,
        "Condition": condition,
        "Brand": brand,
        "Model": title,
        "Storage_GB": 128,
        "Age_years": age,
        "Location": "Kathmandu",
    }


class TestPreprocessingAndTraining(unittest.TestCase):
    def test_preprocessing_cleans_deduplicates_and_derives_features(self):
        rows = [
            listing(" Phone A ", "45,000"),
            listing(" Phone A ", "45,000"),
            listing("Phone B", "invalid"),
        ]
        original = pd.DataFrame(rows)

        cleaned = preprocess_listings(original)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.loc[0, "Title"], "Phone A")
        self.assertEqual(cleaned.loc[0, "Price_NPR"], 45000)
        self.assertEqual(cleaned.loc[0, "Condition_Score"], 8)
        self.assertEqual(cleaned.loc[0, "Age_months"], 12)
        self.assertEqual(cleaned.loc[0, "Price_per_GB"], 351.56)
        self.assertEqual(original.loc[0, "Title"], " Phone A ")

    def test_preprocessing_rejects_missing_schema(self):
        with self.assertRaisesRegex(ValueError, "Dataset is missing required columns"):
            preprocess_listings(pd.DataFrame({"Category": ["Mobile"]}))

    def test_training_persists_reloadable_pipeline_and_metadata(self):
        rows = [
            listing("Phone A", 45000),
            listing("Phone B", 39000, brand="Samsung", condition="Good", age=2),
            listing("Phone C", 32000, brand="Dell", condition="Fair", age=3),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            source_path = temp_path / "source.csv"
            processed_path = temp_path / "processed.csv"
            model_path = temp_path / "model.pkl"
            metadata_path = temp_path / "metadata.json"
            pd.DataFrame(rows).to_csv(source_path, index=False)

            metadata = train_model(
                source=source_path,
                processed_path=processed_path,
                model_path=model_path,
                metadata_path=metadata_path,
            )

            with model_path.open("rb") as model_file:
                model = pickle.load(model_file)
            prediction = model.predict(
                pd.DataFrame(
                    [{
                        "Category": "Mobile",
                        "Condition": "Excellent",
                        "Brand": "TestBrand",
                        "Storage_GB": 128,
                        "Age_years": 1,
                    }]
                )
            )[0]
            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertGreater(prediction, 0)
            self.assertEqual(metadata["training_rows"], 3)
            self.assertIn("Brand", metadata["feature_columns"])
            category_encoder = model.named_steps["preprocessor"].named_transformers_["categories"]
            self.assertIn("TestBrand", category_encoder.categories_[2])
            self.assertEqual(saved_metadata["model_version"], metadata["model_version"])
            self.assertTrue(processed_path.is_file())


if __name__ == "__main__":
    unittest.main()
