"""
Shared test fixtures for RetainML unit tests.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_config():
    """Merged config dict matching ConfigLoader.load_all_configs() output."""
    return {
        "data": {
            "raw_data_path": "data/raw/Telco-Customer-Churn.csv",
            "processed_data_path": "data/processed/",
        },
        "preprocessing": {
            "target_column": "Churn",
            "id_column": "customerID",
            "missing_value_strategy": {
                "TotalCharges": {
                    "method": "fill_with_column",
                    "fill_column": "MonthlyCharges",
                }
            },
            "numeric_conversions": ["TotalCharges"],
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
            "scaler_type": "StandardScaler",
            "apply_smote": False,
            "smote_random_state": 42,
        },
        "feature_engineering": {
            "tenure_groups": {
                "enabled": True,
                "bins": [0, 12, 24, 48, 72],
                "labels": ["0-1 year", "1-2 years", "2-4 years", "4+ years"],
            },
            "derived_features": {
                "charge_ratio": {"enabled": True},
                "avg_monthly_charge": {"enabled": True},
                "high_monthly_charge": {"enabled": True, "threshold_type": "median"},
            },
            "service_features": {
                "enabled": True,
                "service_columns": [
                    "PhoneService", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies",
                ],
                "exclude_values": ["No", "No internet service", "No phone service"],
            },
            "indicator_features": {
                "has_internet": {"enabled": True},
                "has_phone": {"enabled": True},
                "has_family": {"enabled": True},
                "contract_risk": {"enabled": True},
                "payment_risk": {"enabled": True},
                "senior_alone": {"enabled": True},
            },
            "encoding": {
                "binary_encode_yes_no": True,
                "one_hot_encode_categorical": True,
                "drop_first": True,
            },
        },
        "models": {
            "logistic_regression": {
                "enabled": True,
                "display_name": "Logistic Regression",
                "params": {"max_iter": 1000, "random_state": 42, "class_weight": "balanced"},
            },
            "random_forest": {
                "enabled": True,
                "display_name": "Random Forest",
                "params": {"n_estimators": 10, "max_depth": 5, "random_state": 42, "class_weight": "balanced"},
            },
            "gradient_boosting": {
                "enabled": True,
                "display_name": "Gradient Boosting",
                "params": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
            },
        },
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
            "primary_metric": "f1_score",
            "cross_validation": {"enabled": False, "cv_folds": 5, "stratified": True},
            "visualizations": {
                "save_path": "results/",
                "create_model_comparison": True,
                "create_confusion_matrix": True,
                "create_roc_curve": True,
                "create_feature_importance": True,
                "feature_importance_top_n": 20,
                "dpi": 300,
            },
        },
        "pipeline": {
            "name": "test_pipeline",
            "version": "1.0.0",
            "paths": {
                "data_dir": "data/",
                "results_dir": "results/",
                "models_dir": "models/",
                "logs_dir": "logs/",
            },
            "logging": {"level": "WARNING", "log_to_file": False},
        },
        "mlflow": {
            "enabled": False,
            "tracking_uri": "./mlruns",
            "experiment_name": "test_experiment",
            "connection": {"fail_silently": True},
        },
    }


@pytest.fixture
def sample_raw_df():
    """20-row DataFrame mimicking the Telco churn dataset."""
    np.random.seed(42)
    n = 20
    data = {
        "customerID": [f"CUST-{i:04d}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], n),
        "SeniorCitizen": [0]*14 + [1]*6,
        "Partner": ["Yes"]*10 + ["No"]*10,
        "Dependents": ["No"]*12 + ["Yes"]*8,
        "tenure": [6, 15, 36, 60, 1, 24, 48, 3, 12, 70, 5, 30, 45, 55, 8, 20, 35, 50, 2, 65],
        "PhoneService": ["Yes"]*16 + ["No"]*4,
        "MultipleLines": ["Yes"]*6 + ["No"]*8 + ["No phone service"]*4 + ["Yes"]*2,
        "InternetService": ["DSL"]*7 + ["Fiber optic"]*7 + ["No"]*6,
        "OnlineSecurity": ["Yes"]*5 + ["No"]*7 + ["No internet service"]*6 + ["Yes"]*2,
        "OnlineBackup": ["Yes"]*6 + ["No"]*6 + ["No internet service"]*6 + ["No"]*2,
        "DeviceProtection": ["Yes"]*5 + ["No"]*7 + ["No internet service"]*6 + ["Yes"]*2,
        "TechSupport": ["Yes"]*4 + ["No"]*8 + ["No internet service"]*6 + ["No"]*2,
        "StreamingTV": ["Yes"]*6 + ["No"]*6 + ["No internet service"]*6 + ["Yes"]*2,
        "StreamingMovies": ["Yes"]*5 + ["No"]*7 + ["No internet service"]*6 + ["No"]*2,
        "Contract": ["Month-to-month"]*10 + ["One year"]*5 + ["Two year"]*5,
        "PaperlessBilling": ["Yes"]*12 + ["No"]*8,
        "PaymentMethod": ["Electronic check"]*8 + ["Mailed check"]*4 + ["Bank transfer (automatic)"]*4 + ["Credit card (automatic)"]*4,
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89.10, 29.75, 74.40, 85.50,
                          45.20, 61.30, 78.90, 33.15, 95.40, 50.00, 67.80, 40.25, 88.15, 55.60],
        "TotalCharges": ["29.85", "1889.50", "108.15", "1840.75", " ", "2281.30", "3989.40", "29.75",
                        "820.50", "5681.10", "226.00", "1530.50", "3542.15", "1425.60", " ", "960.00",
                        "2109.30", "1810.25", "176.30", "3200.10"],
        "Churn": ["No"]*14 + ["Yes"]*6,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_preprocessed_df(sample_raw_df, sample_config):
    """sample_raw_df after preprocessing and target encoding."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

    from data_pipeline.preprocessors import DataPreprocessor, TargetEncoder

    preprocessor = DataPreprocessor(sample_config)
    df = preprocessor.preprocess(sample_raw_df)

    target_encoder = TargetEncoder(sample_config)
    df = target_encoder.encode_target(df)

    return df


@pytest.fixture
def sample_X_y():
    """Small (X_train, X_test, y_train, y_test) for model tests."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(40, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series([0]*25 + [1]*15, name="Churn")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def fitted_lr_model(sample_X_y):
    """LogisticRegression fitted on sample_X_y train data."""
    X_train, _, y_train, _ = sample_X_y
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


FEATURE_COLUMNS = [
    "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "PaperlessBilling", "MonthlyCharges", "TotalCharges", "charge_ratio",
    "avg_monthly_charge", "high_monthly_charge", "services_count",
    "has_internet", "has_phone", "has_family", "contract_risk", "payment_risk",
    "senior_alone", "gender_Male", "MultipleLines_No phone service",
    "MultipleLines_Yes", "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check", "tenure_group_1-2 years",
    "tenure_group_2-4 years", "tenure_group_4+ years",
]


@pytest.fixture
def feature_columns():
    """The 43 feature column names after encoding."""
    return FEATURE_COLUMNS.copy()


@pytest.fixture
def validation_config():
    """Config dict with validation rules."""
    return {
        "validation": {
            "min_rows": 10,
            "max_missing_pct": 5.0,
            "required_columns": [
                "customerID", "gender", "SeniorCitizen", "Partner",
                "Dependents", "tenure", "PhoneService", "MultipleLines",
                "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV",
                "StreamingMovies", "Contract", "PaperlessBilling",
                "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
            ],
            "id_column": "customerID",
            "target_column": "Churn",
            "target_values": ["Yes", "No"],
            "numeric_ranges": {
                "tenure": {"min": 0, "max": 200},
                "MonthlyCharges": {"min": 0, "max": 500},
            },
            "categorical_values": {
                "Contract": ["Month-to-month", "One year", "Two year"],
                "gender": ["Male", "Female"],
            },
        }
    }


@pytest.fixture
def mock_rf_model(feature_columns):
    """Mock Random Forest model with controlled feature_importances_."""
    np.random.seed(42)
    n = len(feature_columns)
    importances = np.random.dirichlet(np.ones(n))  # sums to 1.0

    model = MagicMock()
    model.feature_importances_ = importances
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture
def mock_scaler(feature_columns):
    """Mock scaler that returns input unchanged."""
    scaler = MagicMock()
    scaler.transform.side_effect = lambda x: x
    return scaler


@pytest.fixture
def sample_customer_data():
    """Single customer raw input dict."""
    return {
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 6,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": "510.0",
        "gender": "Male",
    }
