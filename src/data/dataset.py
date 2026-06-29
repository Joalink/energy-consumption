from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURES = [
    "Household_Size",
    "Avg_Temperature_C",
    "Has_AC",
    "Peak_Hours_Usage_kWh",
    "Month",
    "DayOfWeek",
]
TARGET = "Energy_Consumption_kWh"


class EnergyDataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        path = Path(f"data/splits/{split}.csv")
        data_frame = pd.read_csv(path)
        print(data_frame.isna().sum())
        print(data_frame.columns[data_frame.isna().any()])
        self.X = torch.tensor(data_frame[FEATURES].values, dtype=torch.float32)
        self.y = torch.tensor(data_frame[TARGET].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]
