from pathlib import Path

import torch
from torch.utils.data import DataLoader

import mlflow
from src.config import load_config
from src.data.dataset import EnergyDataset
from src.model.architecture import EnergyModel
from src.training.loss import get_loss
from src.training.optimizer import get_optimizer


def setup_mlflow(config: dict):
    traking_uri = f"sqlite:///{config['paths']['mlflow_db']}"
    mlflow.set_tracking_uri(traking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def val_epoch(model, loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()

    return total_loss / len(loader)


def register_model(config, model, train_loader, val_loader, optimizer, loss_fn, device):
    best_val_loss = float("inf")
    patience_counter = 0

    with mlflow.start_run():
        mlflow.log_params(config)

        for epoch in range(config["training"]["epochs"]):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss = val_epoch(model, val_loader, loss_fn, device)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss},
                step=epoch,
            )

            print(
                f"Epoch {epoch + 1}/{config['training']['epochs']} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                weights_path = Path(config["paths"]["model_output"])
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), weights_path)
                print(f"Best model saved at epoch {epoch + 1}")
            else:
                patience_counter += 1
                if patience_counter >= config["training"]["patience"]:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        mlflow.log_artifact(config["paths"]["model_output"])
        print(f"Training complete. Best val_loss: {best_val_loss:.4f}")


def main() -> None:
    config = load_config()
    setup_mlflow(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device:{device}")

    print("Starting training")
    train_loader = DataLoader(
        EnergyDataset("train"), batch_size=config["training"]["batch"], shuffle=True
    )
    val_loader = DataLoader(
        EnergyDataset("train"),
        batch_size=config["training"]["batch"],
    )

    model = EnergyModel().to(device)
    optimizer = get_optimizer(model)
    loss_fn = get_loss()

    print("Register model")
    register_model(
        config,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
    )
    print("Training complete")


if __name__ == "__main__":
    main()
