import unittest

import app as kinbech_app


class TestKinbechApp(unittest.TestCase):
    def setUp(self):
        kinbech_app.prediction_count = 0
        self.client = kinbech_app.app.test_client()

    def test_pages_use_dataset(self):
        for path in ("/", "/predict", "/dashboard"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)

        self.assertIn(b"500", self.client.get("/").data)
        self.assertIn(b"Laptop", self.client.get("/predict").data)
        self.assertIn(b"Mobile", self.client.get("/predict").data)

    def test_prediction_uses_model_and_shows_comparables(self):
        response = self.client.post(
            "/predict",
            data={
                "category": "Mobile",
                "condition": "Excellent",
                "storage_gb": "256",
                "age_years": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Estimated Listing Price", response.data)
        self.assertIn(b"Apple iPhone 12 256GB", response.data)
        self.assertNotIn(b"<td>Good</td>", response.data)
        self.assertEqual(kinbech_app.prediction_count, 1)

    def test_invalid_prediction_inputs_are_rejected(self):
        valid_data = {
            "category": "Mobile",
            "condition": "Excellent",
            "storage_gb": "256",
            "age_years": "1",
        }
        invalid_inputs = (
            {"category": "Furniture"},
            {"condition": "Perfect"},
            {"storage_gb": "-1"},
            {"age_years": "-1"},
            {"storage_gb": "nan"},
            {"age_years": "inf"},
        )

        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                response = self.client.post(
                    "/predict",
                    data={**valid_data, **invalid_input},
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn(b"role=\"alert\"", response.data)

        self.assertEqual(kinbech_app.prediction_count, 0)

    def test_price_distribution_returns_dataset_prices(self):
        response = self.client.get("/api/price-distribution")

        self.assertEqual(response.status_code, 200)
        prices = response.get_json()["prices"]
        self.assertEqual(len(prices), len(kinbech_app.listings))
        self.assertGreater(min(prices), 0)


if __name__ == "__main__":
    unittest.main()
