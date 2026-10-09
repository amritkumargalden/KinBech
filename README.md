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
