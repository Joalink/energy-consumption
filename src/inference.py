from pathlib import Path

import torch

WEIGHTS_PATH = Path("model/best.pt")


class EnergyInference:
    def __init__(self) -> None:
        self.service = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EnergyInference().to(self.device)
        self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
        self.model.eval()

    def predict(self, features: list[float]) -> float:
        tensor = torch.tensor([features], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return float(self.model(tensor).squeeze())
