# Surprise Housing India

An AI-powered real estate valuation and investment decision-support dashboard
for the Indian housing market. The application is built with Streamlit and
uses four trained regression models to estimate property values.

## Live application

[Open the deployed Streamlit dashboard](https://kavaishnavi7-suprise-housing-india-app-v8oibo.streamlit.app/)

## Features

- Property valuation in Indian rupees using XGBoost as the primary model
- Side-by-side predictions from Lasso, Ridge, Random Forest, and XGBoost
- Estimated price per square foot and valuation range
- Investment rating based on RERA and Vastu inputs
- Model benchmark metrics including R², MAE, RMSE, and MAPE
- Lasso-based positive and negative price-driver visualizations
- Interactive price sensitivity simulator
- Inputs for metro city, locality tier, area, BHK, possession, amenities,
  connectivity, parking, and building details

## Project structure

```text
.
├── app.py                 # Streamlit dashboard
├── train.py               # Dataset generation, training, and artifact export
├── generate_dataset.py    # Synthetic Indian housing dataset generator
├── src/
│   ├── models.py          # Model training and evaluation helpers
│   └── preprocessing.py   # Feature engineering and scaling
├── data/train.csv         # Training dataset
├── models/                # Saved models, scaler, and metadata
└── requirements.txt       # Python dependencies
```

## Run locally

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` by default.

## Retrain the models

The trained artifacts are included in `models/`. To regenerate the dataset
and retrain all models:

```powershell
python train.py
```

This updates the model files, metadata, metrics, and training CSV used by the
dashboard.

## Deploy with Streamlit Community Cloud

1. Fork or connect this repository in [Streamlit Community Cloud](https://share.streamlit.io/).
2. Select the `main` branch.
3. Set the main file path to `app.py`.
4. Deploy.

No secrets are required for the current dashboard. A Google Maps API key
should only be added through Streamlit Secrets if map functionality is enabled;
never commit API keys to this repository.

## Model notes

The target is modeled in log-price space and converted back to INR for display.
The dataset is generated for demonstration and prototyping purposes. Valuations
are estimates, not certified appraisals or financial advice.

## License

This project is provided for educational and demonstration purposes.
