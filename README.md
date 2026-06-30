# Energy Consumption Predictor

Machine learning API that predicts household energy consumption in kWh based on household characteristics and environmental conditions.


## Model

- Architecture: MLP (Multilayer Perceptron) trained from scratch with PyTorch
- Task: Regression — predicts energy consumption in kWh
- Input features: 6
- Training: Early stopping with patience 10, Adam optimizer
- Metrics on test set:
  - MAE: 0.5193 kWh
  - RMSE: 0.6704 kWh
  - MAPE: 6.60%
  - R2: 0.9851

### Request fields (classes)

| Field | Type | Description |
|-------|------|-------------|
| household_size | int | Number of people in the household |
| avg_temperature_c | float | Average daily temperature in Celsius |
| has_ac | bool | Whether the household has air conditioning |
| peak_hours_usage_kwh | float | Energy used during peak hours |
| month | int | Month of the year (1-12) |
| day_of_week | int | Day of the week (0=Monday, 6=Sunday) |


## Structured Json

```json
{
    "household_size": 4,
    "avg_temperature_c": 28.5,
    "has_ac": true,
    "peak_hours_usage_kwh": 3.2,
    "month": 7,
    "day_of_week": 2
}

```

Send household data, get back a predicted energy consumption value.

```json
{
  "energy_consumption_kwh": 24.73
}
```




## Tech stack

| Layer | Technology |
|-------|------------|
| Model | PyTorch MLP |
| Experiment tracking | MLflow |
| Pipeline orchestration | Airflow |
| API | FastAPI |
| Containerization | Docker |
| Dependency management | uv |
| Deployment | Railway |
| Dev environment | Dev Containers |

## Project structure

```
energy-consumption/
│
├── src/
│   ├── data/
│   │   ├── collect.py        # downloads raw data
│   │   ├── clean.py          # cleans, engineers features, splits data
│   │   └── dataset.py        # PyTorch Dataset class
│   │
│   ├── model/
│   │   └── architecture.py   # MLP definition
│   │
│   ├── training/
│   │   ├── trainer.py        # training loop with MLflow logging
│   │   ├── optimizer.py      # Adam optimizer setup
│   │   └── loss.py           # MSELoss
│   │
│   ├── evaluation/
│   │   └── metrics.py        # MAE, RMSE, MAPE, R2 on test set
│   │
│   ├── config.py             # reads model config
│   └── inference.py          # loads weights, runs predictions
│
├── api/
│   ├── main.py               # FastAPI endpoints
│   └── schemas.py            # Pydantic input/output types
│
├── dags/
│   └── training_dag.py       # Airflow pipeline
│
├── configs/
│   └── training_config.yaml  # training hyperparameters and architecture parameters
│
├── models/
│   └── best.pt               # trained model weights
│
├── Dockerfile
├── railway.toml
├── pyproject.toml
└── README.md
```

## Local setup

Requirements: Docker, VS Code, Dev Containers extension.

```bash
# clone the repo
git clone https://github.com/your-username/energy-consumption
cd energy-consumption

# open in dev container
# VS Code will prompt automatically — click Reopen in Container

# download the dataset for energy consumption
https://www.kaggle.com/datasets/mrsimple07/energy-consumption-prediction?resource=download

# Create data directories
mkdir data/raw
mkdir data/processed
mkdir data/splits

# locate the dataset on data/raw

# run the full pipeline
uv run python -m src.data.collect
uv run python -m src.data.clean
uv run python -m src.training.trainer
uv run python -m src.evaluation.metrics

# start the API
uv run python -m uvicorn api.main:app --reload
```

## MLflow tracking

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000` to view experiments, metrics, and model versions.

## Pipeline order

```
collect → clean → train → evaluate → deploy
```

Managed by Airflow. Trigger manually:

```bash
uv run airflow dags trigger energy_training_pipeline
```

## Environment variables

| Variable | Description |
|----------|-------------|
| PYTHONPATH | Set to /app in production |

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Upload energy data to analize |

## Deployment

This service is deployed on Railway

Base URL: `https://energy-consumption-production.up.railway.app/predict`

Interactive docs: `https://energy-consumption-production.up.railway.app/docs`

## License

MIT

