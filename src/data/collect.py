from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("data/raw")


def main() -> None:
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    data_frame = pd.read_csv(RAW_DATA_PATH / "household_energy_consumption.csv")
    data_frame.to_csv(RAW_DATA_PATH / "energy.csv", index=False)
    print(f"Collected {len(data_frame)} frames")


if __name__ == "__main__":
    main()
