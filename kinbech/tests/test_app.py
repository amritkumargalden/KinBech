import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import app as kinbech_app


class TestKinbechApp(unittest.TestCase):
    def setUp(self):
        kinbech_app.prediction_count = 0
        self.temp_dir = tempfile.TemporaryDirectory()
        kinbech_app.app.config['PREDICTION_LOG_PATH'] = str(
            Path(self.temp_dir.name) / "prediction_events.sqlite3"
        )
        self.addCleanup(self.temp_dir.cleanup)
        self.client = kinbech_app.app.test_client()

    def _read_event(self, query):
        connection = sqlite3.connect(kinbech_app.app.config['PREDICTION_LOG_PATH'])
        try:
            return connection.execute(query).fetchone()
        finally:
            connection.close()

    def test_prediction_options_return_model_categories_and_conditions(self):
        response = self.client.get(
            "/api/options",
            headers={"Origin": "http://127.0.0.1:8000"},
        )

        self.assertEqual(response.status_code, 200)
        options = response.get_json()
        self.assertEqual(options["categories"], ["Laptop", "Mobile"])
        self.assertIn("Excellent", options["conditions"])
        self.assertIn("Apple", options["brands"])
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
                "brand": "Apple",
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
        event = self._read_event(
            "SELECT source, model_version, request_json, result_json FROM prediction_events"
        )
        self.assertEqual(event[0], "api")
        self.assertEqual(event[1], kinbech_app.MODEL_VERSION)
        self.assertEqual(json.loads(event[2])["Original_Price_NPR"], 45000)
        self.assertEqual(
            json.loads(event[3])["predicted_price"],
            result["predicted_price"],
        )

    def test_brand_changes_the_price_estimate(self):
        estimates = {}
        for brand in ("Apple", "Xiaomi"):
            response = self.client.post(
                "/api/predict",
                json={
                    "category": "Mobile",
                    "condition": "Excellent",
                    "brand": brand,
                    "storage_gb": 256,
                    "age_years": 1,
                },
            )
            self.assertEqual(response.status_code, 200)
            estimates[brand] = response.get_json()["predicted_price"]

        self.assertNotEqual(estimates["Apple"], estimates["Xiaomi"])

    def test_listing_price_analysis_flags_low_and_high_outliers(self):
        low = kinbech_app.analyze_listing_price(
            "Mobile", "Apple", "Excellent", 1
        )
        high = kinbech_app.analyze_listing_price(
            "Mobile", "Apple", "Excellent", 10_000_000
        )
        typical = kinbech_app.analyze_listing_price(
            "Mobile", "Apple", "Excellent", 100_000
        )

        self.assertEqual(low["status"], "unusually_low")
        self.assertEqual(high["status"], "unusually_high")
        self.assertEqual(typical["status"], "within_range")
        self.assertEqual(typical["comparison_level"], "category_brand_condition")
        self.assertGreaterEqual(typical["sample_count"], 5)
        self.assertIn("minimum", typical["reference_range"])
        self.assertIn("maximum", typical["reference_range"])

    def test_listing_price_analysis_falls_back_and_handles_insufficient_data(self):
        fallback = kinbech_app.analyze_listing_price(
            "Mobile", "Apple", "missing condition", 100_000
        )
        self.assertEqual(fallback["comparison_level"], "category_brand")
        self.assertEqual(fallback["comparison_basis"], "Category and brand")

        with patch.object(
            kinbech_app,
            "DATASET",
            kinbech_app.DATASET[kinbech_app.DATASET["Category"] == "no category"],
        ):
            insufficient = kinbech_app.analyze_listing_price(
                "no category", "Apple", "Excellent", 100_000
            )

        self.assertEqual(insufficient["status"], "insufficient_data")
        self.assertEqual(insufficient["sample_count"], 0)

    def test_listing_price_analysis_uses_age_matched_peers_then_falls_back(self):
        peers = kinbech_app.pd.DataFrame({
            "Category": ["Mobile"] * 5,
            "Brand": ["Apple"] * 5,
            "Condition": ["Excellent"] * 5,
            "Price_NPR": [80_000, 85_000, 90_000, 95_000, 100_000],
            "Age_years": [2.0, 2.2, 2.4, 2.6, 2.8],
        })

        with patch.object(kinbech_app, "DATASET", peers):
            age_matched = kinbech_app.analyze_listing_price(
                "Mobile", "Apple", "Excellent", 90_000, age_years=2.5
            )
            fallback = kinbech_app.analyze_listing_price(
                "Mobile", "Apple", "Excellent", 90_000, age_years=10
            )

        self.assertEqual(
            age_matched["comparison_level"], "category_brand_condition_age"
        )
        self.assertEqual(age_matched["sample_count"], 5)
        self.assertEqual(fallback["comparison_level"], "category_brand_condition")
        self.assertEqual(fallback["sample_count"], 5)

    def test_batch_listing_price_analysis_validates_batches_and_items(self):
        response = self.client.post(
            "/api/analyze-listing-prices",
            json={"listings": [
                {
                    "id": 7,
                    "category": "Mobile",
                    "brand": "Apple",
                    "condition": "Excellent",
                    "age_years": 1,
                    "price": 100_000,
                },
                {
                    "id": 8,
                    "category": "Mobile",
                    "brand": "not-supported",
                    "condition": "Excellent",
                    "age_years": 1,
                    "price": 100_000,
                },
                {
                    "id": 9,
                    "category": "Mobile",
                    "brand": "Apple",
                    "condition": "Excellent",
                    "age_years": True,
                    "price": True,
                },
                None,
            ]},
        )

        self.assertEqual(response.status_code, 200)
        results = response.get_json()["results"]
        self.assertEqual(results[0]["id"], 7)
        self.assertIn(
            results[0]["status"],
            {"unusually_low", "unusually_high", "within_range"},
        )
        self.assertEqual(results[1]["status"], "unassessed")
        self.assertEqual(results[1]["sample_count"], 0)
        self.assertEqual(results[2]["status"], "unassessed")
        self.assertEqual(results[2]["sample_count"], 0)
        self.assertEqual(results[3]["status"], "unassessed")
        self.assertEqual(results[3]["sample_count"], 0)

    def test_batch_listing_price_analysis_rejects_missing_or_oversized_batches(self):
        for payload in ({}, {"listings": []}, {"listings": [None] * 51}):
            with self.subTest(payload_size=len(payload.get("listings", []))):
                response = self.client.post(
                    "/api/analyze-listing-prices", json=payload
                )
                self.assertEqual(response.status_code, 400)

    def test_prediction_api_returns_optional_listing_price_analysis(self):
        response = self.client.post(
            "/api/predict",
            json={
                "category": "Mobile",
                "condition": "Excellent",
                "brand": "Apple",
                "storage_gb": 256,
                "age_years": 1,
                "listing_price": 1,
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["listing_price"], 1)
        self.assertEqual(
            result["listing_price_analysis"]["status"], "unusually_low"
        )
        event = self._read_event(
            "SELECT request_json, result_json FROM prediction_events"
        )
        self.assertEqual(json.loads(event[0])["Listing_Price_NPR"], 1)
        self.assertEqual(
            json.loads(event[1])["listing_price_analysis"]["status"],
            "unusually_low",
        )

    def test_storefront_prediction_api_rejects_invalid_values(self):
        response = self.client.post(
            "/api/predict",
            json={
                "category": "Furniture",
                "condition": "Perfect",
                "brand": "not-a-listed-brand",
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
        self.assertIn(b'name="listing_price"', response.data)
        self.assertNotIn(b'type="range"', response.data)

        response = self.client.post(
            "/predict",
            data={
                "category": "Mobile",
                "brand": "Apple",
                "condition": "Excellent",
                "storage_gb": "256",
                "age_years": "1",
                "original_price": "45000",
                "listing_price": "1",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Estimated Listing Price", response.data)
        self.assertIn(b"Apple", response.data)
        self.assertIn(b"Price outlier to review", response.data)
        self.assertIn(b"unusually low", response.data)
        event = self._read_event(
            "SELECT source, request_json FROM prediction_events"
        )
        self.assertEqual(event[0], "form")
        self.assertEqual(json.loads(event[1])["Storage_GB"], 256.0)

    def test_dashboard_metrics_and_health_check(self):
        dashboard_response = self.client.get("/dashboard")
        stats_response = self.client.get("/api/price-stats")
        health_response = self.client.get("/health")

        self.assertEqual(dashboard_response.status_code, 200)
        self.assertIn(b"Market Trends Dashboard", dashboard_response.data)
        self.assertIn(b"Median price by category", dashboard_response.data)
        self.assertIn(b"Popular brands", dashboard_response.data)
        self.assertIn(b"Price by item age", dashboard_response.data)
        self.assertIn(b"Price by condition", dashboard_response.data)
        self.assertIn(b"Popular locations", dashboard_response.data)
        self.assertIn(b"no listing dates", dashboard_response.data)
        self.assertEqual(stats_response.status_code, 200)
        self.assertEqual(stats_response.get_json()["count"], len(kinbech_app.DATASET))
        self.assertGreater(stats_response.get_json()["mean"], 0)
        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(health_response.get_json()["status"], "healthy")
        self.assertTrue(health_response.get_json()["model_loaded"])
        self.assertTrue(health_response.get_json()["dataset_loaded"])

    def test_market_trend_aggregates_and_empty_dataset(self):
        trends = kinbech_app.build_market_trends(kinbech_app.DATASET)

        self.assertEqual(trends["stats"]["total_listings"], len(kinbech_app.DATASET))
        self.assertGreater(trends["stats"]["median_price"], 0)
        self.assertTrue(trends["category_trends"])
        self.assertLessEqual(len(trends["brand_trends"]), 8)
        self.assertTrue(trends["condition_trends"])
        self.assertTrue(trends["age_trends"])
        self.assertTrue(trends["location_trends"])
        self.assertTrue(trends["insights"])
        for trend_list in (
            trends["category_trends"],
            trends["brand_trends"],
            trends["condition_trends"],
            trends["age_trends"],
            trends["location_trends"],
        ):
            for trend in trend_list:
                self.assertGreater(trend["listings"], 0)
                self.assertGreater(trend["median_price"], 0)
                self.assertGreater(trend["bar_pct"], 0)
                self.assertLessEqual(trend["bar_pct"], 100)

        empty = kinbech_app.build_market_trends(kinbech_app.pd.DataFrame())
        self.assertEqual(empty["stats"]["total_listings"], 0)
        self.assertEqual(empty["category_trends"], [])

    def test_health_check_reports_unavailable_dependencies(self):
        with patch.object(kinbech_app, "MODEL", None), patch.object(
            kinbech_app, "DATASET", kinbech_app.pd.DataFrame()
        ):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["status"], "unhealthy")

    def test_prediction_is_not_reported_as_success_when_logging_fails(self):
        with patch.object(
            kinbech_app,
            "log_prediction",
            side_effect=kinbech_app.PredictionLogError("database unavailable"),
        ), patch.object(kinbech_app.logger, "exception"):
            response = self.client.post(
                "/api/predict",
                json={
                    "category": "Mobile",
                    "condition": "Excellent",
                    "brand": "Apple",
                    "storage_gb": 256,
                    "age_years": 1,
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("could not be recorded", response.get_json()["error"])
        self.assertEqual(kinbech_app.prediction_count, 0)


if __name__ == "__main__":
    unittest.main()
