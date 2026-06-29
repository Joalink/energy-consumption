from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EnergyDataset
from src.model.architecture import EnergyModel

WEIGHTS_PATH = Path("models/best.pt")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EnergyModel().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    loader = DataLoader(EnergyDataset("test"), batch_size=32)

    preds, targets = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100
    r2 = 1 - np.sum((targets - preds) ** 2) / np.sum((targets - np.mean(targets)) ** 2)

    print(f"MAE:  {mae:.4f} kWh")
    print(f"RMSE: {rmse:.4f} kWh")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2:   {r2:.4f}")


if __name__ == "__main__":
    main()
