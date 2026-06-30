# Changelog

All notable changes to this project will be documented here.

## [1.0.0] - 2026-06-30

### Added
- Initial MLP model trained from scratch with PyTorch
- Data pipeline: collect, clean, feature engineering, train/val/test split
- MLflow experiment tracking with parameter and metric logging
- Airflow DAG for full pipeline orchestration with quality gate
- FastAPI REST API with /health and /predict endpoints
- Docker containerization for production
- Railway deployment with shell-wrapped start command for $PORT expansion
- Dev container setup with GPU support and uv dependency management
- Early stopping based on validation loss
- Evaluation metrics: MAE, RMSE, MAPE, R2

### Model
- Architecture: MLP with 2 hidden layers
- Input: 6 features (Household_Size, Avg_Temperature_C, Has_AC, Peak_Hours_Usage_kWh, Month, DayOfWeek)
- Output: Energy consumption in kWh
- Optimizer: Adam
- Loss: MSELoss