# KinBech

KinBech estimates second-hand item prices from the CSV dataset in `Data`.
The Flask application and its trained model live in `kinbech`.

## Run on Windows

From the project root, create and prepare the environment once:

```powershell
py -3.12 -m venv kinbech_env
.\kinbech_env\Scripts\python.exe -m pip install -r requirements.txt
```

Train or rebuild the model from the current CSV, then start the web app:

```powershell
.\kinbech_env\Scripts\python.exe .\Data\model_training.py
.\kinbech_env\Scripts\python.exe .\kinbech\app.py
```

Open <http://127.0.0.1:5000/>. The app reads the CSV directly; MySQL is not
required to browse listings, make predictions, or view the dashboard.

## Run tests

```powershell
Set-Location .\kinbech
..\kinbech_env\Scripts\python.exe -m unittest discover -s tests -v
```

The server is configured for local development. Do not expose Flask's debug
server to the public internet.

## Run the EverSales + KinBech experience

EverSales remains the marketplace frontend and PHP/MySQL backend. Its
**Price Predictor** page calls the KinBech model service for category options
and predictions. From the project root, start each local development server:

```powershell
php -S 127.0.0.1:8000 -t .\eversales
.\.venv\Scripts\python.exe .\kinbech\app.py
```

Open <http://127.0.0.1:8000/frontend/index.html> and choose **Price Predictor**.
The predictor defaults to the KinBech API at `http://127.0.0.1:5000`; update
the `kinbech-api-base` meta value in
`eversales/frontend/price-predictor.html` when the model service uses a
different address. For a different EverSales web origin, set
`EVERSALES_ORIGINS` to a comma-separated list of allowed origins before
starting KinBech. The local predictor API does not use credentials; configure
HTTPS and a production origin policy before deploying it.
