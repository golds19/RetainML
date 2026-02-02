@echo off
echo ============================================
echo  RetainML - Starting MLflow Tracking UI
echo ============================================
echo.
echo Tracking URI: sqlite:///mlflow.db (local SQLite database)
echo UI will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server.
echo.

cd /d "%~dp0\.."
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
