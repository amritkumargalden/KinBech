# tests/test_model.py
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        
    def test_prediction_output_type(self):
        X_test = np.array([[50000, 12, 7, 0.5, 0.7]])
        self.model.fit(X_test, np.array([35000]))
        prediction = self.model.predict(X_test)
        self.assertIsInstance(prediction[0], (int, float, np.number))
        
    def test_prediction_range(self):
        X_test = np.array([[50000, 12, 7, 0.5, 0.7]])
        y_train = np.array([35000])
        self.model.fit(X_test, y_train)
        prediction = self.model.predict(X_test)[0]
        self.assertGreater(prediction, 0)

if __name__ == '__main__':
    unittest.main()