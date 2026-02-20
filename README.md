# RetainML: MLOps Customer Churn Prediction System

An end-to-end MLOps platform for predicting customer churn with AI-powered retention recommendations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                                │
│  CSV → DataLoader → DataPreprocessor → ChurnFeatureEngineer         │
│       → DataSplitter → FeatureEncoder (fit/train only)             │
│       → FeatureScaler → ImbalanceHandler (SMOTE)                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                        Training                                     │
│  ModelFactory → ModelTrainer → ModelEvaluator                       │
│  HyperparameterTuner (Optuna, --tune flag)                         │
│  MLflowTracker (experiment tracking + model registry)              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  model.pkl / scaler.pkl / feature_columns.json
┌───────────────────────────────▼─────────────────────────────────────┐
│                        Serving                                      │
│  FastAPI → ChurnPredictor → ChurnExplainer                         │
│  PredictionLogger (SQLite, WAL mode, drift detection)              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                        Dashboard                                    │
│  Streamlit → /predict endpoint → /history endpoint                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Model Performance

Best model trained on the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset:

| Model             | F1-Score | ROC-AUC |
|-------------------|----------|---------|
| Random Forest     | 0.631    | 0.841   |

Run `python scripts/train_pipeline.py --tune` to retrain with Optuna HPO and potentially improve these numbers.

## Key Features

- **Leakage-free pipeline**: `FeatureEncoder` is fit on training data only; the `high_monthly_charge` threshold is computed from training data only and applied to the test set
- **Hyperparameter optimization**: Optuna-powered HPO activated with `--tune`; best params logged to MLflow and saved to `results/hpo/`
- **Explainability**: feature-importance-driven retention plan generator maps model signals to business actions
- **Observability**: every `/predict` call is logged to SQLite; `get_distribution_stats()` detects probability drift against the training baseline
- **Config-driven**: all model params, HPO search spaces, risk thresholds, and feature flags live in `config/`

## How to Run

### Local

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python scripts/train_pipeline.py            # baseline training
python scripts/train_pipeline.py --tune     # with Optuna HPO (slower)

# 4. Start the API (from project root)
cd src && uvicorn serving.app:app --reload --port 8000

# 5. Start the dashboard (separate terminal)
cd src && streamlit run dashboard/app.py
```

API docs available at `http://localhost:8000/docs`.

### Docker

```bash
# Build and start all services
docker compose up --build

# Services:
#   API:       http://localhost:8000
#   Dashboard: http://localhost:8501
#   MLflow UI: http://localhost:5000
```

### Run tests

```bash
# Unit tests
cd src && PYTHONPATH=. python -m pytest tests/unit/ -v

# Integration tests (requires trained model)
cd src && PYTHONPATH=. python -m pytest tests/integration/ -v
```

## Project Structure

```
RetainML/
├── config/                  # YAML configs (models, features, HPO, risk thresholds)
├── data/                    # Raw and processed data
├── docker/                  # Dockerfiles for API and dashboard
├── models/                  # Saved model artifacts (model.pkl, scaler.pkl, ...)
├── results/                 # Evaluation plots, CSV results, HPO outputs
├── scripts/
│   └── train_pipeline.py    # End-to-end training script
└── src/
    ├── data_pipeline/       # Loaders, preprocessors, feature engineering, splitters
    ├── training/            # Models, trainers, evaluators, HPO tuner
    ├── serving/             # FastAPI app, predictor, explainer
    ├── monitoring/          # MLflow tracker, prediction logger (drift detection)
    ├── dashboard/           # Streamlit app
    └── tests/
        ├── unit/            # Unit tests (run in CI)
        └── integration/     # End-to-end tests (require trained model)
```

## License

This project is licensed under the MIT License.
