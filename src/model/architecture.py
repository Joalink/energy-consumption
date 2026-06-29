import torch
import torch.nn as nn

from src.config import load_config


class EnergyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        config = load_config()

        self.network = nn.Sequential(
            nn.Linear(config["model"]["input_size"], config["model"]["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(config["model"]["dropout"]),
            nn.Linear(
                config["model"]["hidden_size"], config["model"]["hidden_size"] // 2
            ),
            nn.ReLU(),
            nn.Dropout(config["model"]["dropout"]),
            nn.Linear(config["model"]["hidden_size"] // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()
