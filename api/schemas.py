from pydantic import BaseModel


class PredictRequest(BaseModel):
    household_size: int
    avg_temperature_c: float
    has_ac: bool
    peak_hours_usage_kwh: float
    month: int
    day_of_week: int


class PredictionResponse(BaseModel):
    energy_consumption_kwh: float
