import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import app as kinbech_app


class TestKinbechApp(unittest.TestCase):
    def setUp(self):
        kinbech_app.prediction_count = 0
        self.client = kinbech_app.app.test_client()

    def test_prediction_options_return_model_categories_and_conditions(self):
        response = self.client.get(
            "/api/options",
            headers={"Origin": "http://127.0.0.1:8000"},
        )

        self.assertEqual(response.status_code, 200)
        options = response.get_json()
        self.assertEqual(options["categories"], ["Laptop", "Mobile"])
        self.assertIn("Excellent", options["conditions"])
        self.assertEqual(
            response.headers["Access-Control-Allow-Origin"],
            "http://127.0.0.1:8000",
        )

    def test_storefront_prediction_api_returns_estimate_and_comparables(self):
        response = self.client.post(
            "/api/predict",
            json={
                "category": "Mobile",
                "condition": "Excellent",
                "storage_gb": 256,
                "age_years": 1,
                "original_price": 45000,
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertGreater(result["predicted_price"], 0)
        self.assertEqual(result["depreciation_pct"], round(
            ((45000 - result["predicted_price"]) / 45000) * 100, 2
        ))
        self.assertTrue(result["comparables"])
        self.assertIn("title", result["comparables"][0])
        self.assertEqual(kinbech_app.prediction_count, 1)

    def test_storefront_prediction_api_rejects_invalid_values(self):
        response = self.client.post(
            "/api/predict",
            json={
                "category": "Furniture",
                "condition": "Perfect",
                "storage_gb": "nan",
                "age_years": 1,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("validation_errors", response.get_json())
        self.assertEqual(kinbech_app.prediction_count, 0)

    def test_kinbech_form_uses_training_condition_labels(self):
        response = self.client.get("/predict")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'value="Like New"', response.data)
        self.assertNotIn(b'type="range"', response.data)

        response = self.client.post(
            "/predict",
            data={
                "category": "Mobile",
                "condition": "Excellent",
                "storage_gb": "256",
                "age_years": "1",
                "original_price": "45000",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Estimated Listing Price", response.data)


if __name__ == "__main__":
    unittest.main()
