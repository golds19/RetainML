"""
Test MLflow connection and configuration.
Run this before enabling MLflow in the pipeline.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.config_loader import ConfigLoader


def test_mlflow_import():
    """Test that mlflow is installed."""
    try:
        import mlflow
        print(f"[PASS] MLflow version: {mlflow.__version__}")
        return True
    except ImportError:
        print("[FAIL] MLflow is not installed. Run: pip install mlflow")
        return False


def test_config_loaded():
    """Test that mlflow_config.yaml loads correctly."""
    config_loader = ConfigLoader(config_dir='config')
    config = config_loader.load_all_configs()
    mlflow_config = config.get('mlflow', {})

    if not mlflow_config:
        print("[WARN] No mlflow config found. Create config/mlflow_config.yaml")
        return False

    print(f"\n[PASS] MLflow config loaded:")
    print(f"  enabled: {mlflow_config.get('enabled')}")
    print(f"  tracking_uri: {mlflow_config.get('tracking_uri')}")
    print(f"  experiment_name: {mlflow_config.get('experiment_name')}")
    return True


def test_tracking_uri():
    """Test that the configured tracking URI is reachable."""
    import mlflow

    config_loader = ConfigLoader(config_dir='config')
    config = config_loader.load_all_configs()
    mlflow_config = config.get('mlflow', {})

    uri = mlflow_config.get('tracking_uri', './mlruns')
    print(f"\nTracking URI: {uri}")

    try:
        mlflow.set_tracking_uri(uri)
        experiment_name = mlflow_config.get('experiment_name', 'churn_prediction')
        mlflow.set_experiment(experiment_name)

        # Try a test run
        with mlflow.start_run(run_name="connection_test") as run:
            mlflow.log_param("test_key", "test_value")
            mlflow.log_metric("test_metric", 1.0)
            run_id = run.info.run_id

        # Clean up test run
        mlflow.delete_run(run_id)

        print(f"[PASS] Successfully connected and created test run")
        print(f"[PASS] Experiment: {experiment_name}")
        return True

    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("RetainML - MLflow Connection Test")
    print("=" * 50)

    results = []
    results.append(("Import", test_mlflow_import()))
    results.append(("Config", test_config_loaded()))
    results.append(("Connection", test_tracking_uri()))

    print("\n" + "=" * 50)
    print("Summary:")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed. MLflow is ready to use.")
    else:
        print("\nSome tests failed. Fix issues before enabling MLflow.")

    sys.exit(0 if all_pass else 1)
