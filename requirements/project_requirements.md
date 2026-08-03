# Pearls AQI Predictor - Project Requirements

## Project Overview
Let's predict the Air Quality Index (AQI) in your city in the next 3 days, using a 100% serverless stack. This project involves building an end-to-end machine learning pipeline for AQI forecasting with automated data collection, feature engineering, model training, and real-time predictions through a web dashboard.
- **Reference Project Description**: [Project Description PDF](https://drive.google.com/file/d/1HPf17hvqI6icNTjRPkPuydkV1ub_lxO5/view?usp=sharing)

---

## Technology Stack

The project utilizes the following technologies and tools:
- **Core Programming**: Python
- **Machine Learning**: Scikit-learn, TensorFlow (or PyTorch)
- **Feature Store & Model Registry**: Hopsworks or Vertex AI
- **Orchestration & CI/CD**: Apache Airflow or GitHub Actions
- **Web UI & Dashboards**: Streamlit, Gradio
- **Backend APIs**: Flask, FastAPI
- **Weather & AQI Data APIs**: AQICN or OpenWeather APIs
- **Model Explainability**: SHAP, LIME
- **Version Control**: Git

---

## Key Features

### 1. Feature Pipeline Development
- **API Integration**: Fetch raw weather and pollutant data from external APIs like AQICN or OpenWeather.
- **Feature Engineering**: Compute features from raw data:
  - Time-based features (hour, day, month).
  - Derived features (such as AQI change rate).
- **Feature Store Integration**: Store processed features in Hopsworks or Vertex AI.

### 2. Historical Data Backfill
- Run the feature pipeline on past dates to generate training data.
- Create a comprehensive, unified dataset for model training and evaluation.

### 3. Training Pipeline Implementation
- Fetch historical features and targets from the Feature Store.
- Train and experiment with various machine learning models (e.g., Random Forest, Ridge Regression, TensorFlow/PyTorch deep learning models).
- Evaluate model performance using standard metrics (RMSE, MAE, and $R^2$ coefficient).
- Register and store trained models in the Model Registry.

### 4. Automated CI/CD Pipeline
- **Feature Pipeline**: Runs automatically every hour to ingest fresh weather and pollutant data.
- **Training Pipeline**: Runs daily to update and re-train models on recent data.
- **Automation Tools**: Use Apache Airflow, GitHub Actions, or similar automated schedulers.

### 5. Web Application Dashboard
- Load trained models and feature sets from the Feature Store / Model Registry.
- Compute real-time forecasts/predictions for the next 3 days.
- Build and display an interactive user dashboard using Streamlit/Gradio and Flask/FastAPI.

### 6. Advanced Analytics Features
- **Exploratory Data Analysis (EDA)**: Identify trends, seasonality, and correlations in the data.
- **Explainability & Interpretation**: Use SHAP or LIME for visual feature importance explanations.
- **Alerting System**: Implement notifications/alerts for hazardous AQI levels.
- **Multi-Model Support**: Support multiple forecasting models (from statistical/regression models to deep learning neural networks).

---

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| Feature pipeline (API fetch + engineering) | Done | `src/features/`, `pipelines/feature_pipeline.py` |
| Historical backfill | Done | `historical_backfill.py` |
| Hopsworks Feature Store upload | Done | `upload_to_hopsworks.py`, `src/features/pipeline.py` |
| Training pipeline (Random Forest) | Done | `src/training/train.py`, `pipelines/training_pipeline.py` |
| Model Registry (Hopsworks) | Done | `src/training/train.py` |
| GitHub Actions (hourly + daily) | Done | `.github/workflows/` |
| Streamlit dashboard | Done | `app/streamlit_app.py` |
| Flask REST API | Done | `app/flask_api.py` |
| EDA | Done | Streamlit → EDA page |
| SHAP explainability | Done | Streamlit → SHAP page |
| Hazardous AQI alerts | Done | `src/alerts.py`, dashboard + API |
| 3-day forecast | Done | `src/inference/predict.py` |

### Quick Start

```bash
pip install -r requirements/requirements.txt
cp .env.example .env   # add API keys

# 1. Backfill historical data
python historical_backfill.py

# 2. Upload to Hopsworks (optional)
python upload_to_hopsworks.py

# 3. Train models
python pipelines/training_pipeline.py

# 4. Run hourly feature pipeline
python pipelines/feature_pipeline.py

# 5. Launch dashboard
streamlit run app/streamlit_app.py

# 6. Launch API
python app/flask_api.py
```
