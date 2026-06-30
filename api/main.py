from functools import lru_cache

from fastapi import FastAPI, HTTPException

from api.schemas import PredictRequest, PredictResponse
from src.inference import EnergyInference

app = FastAPI(title="Energy Consumption API", version="1.0.0")


@lru_cache(maxsize=1)
def get_model() -> EnergyInference:
    return EnergyInference()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        model = get_model()
        features = [
            request.household_size,
            request.avg_temperature_c,
            int(request.has_ac),
            request.peak_hours_usage_kwh,
            request.month,
            request.day_of_week,
        ]
        prediction = model.predict(features)
        return PredictResponse(energy_consumption_kwh=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
