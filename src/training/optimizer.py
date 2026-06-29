import torch.optim as optim

from src.config import load_config


def get_optimizer(model):
    config = load_config()
    return optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
