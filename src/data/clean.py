from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
SPLITS_DATA_PATH = Path("data/splits")


def main() -> None:
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    SPLITS_DATA_PATH.mkdir(parents=True, exist_ok=True)

    data_frame = pd.read_csv(RAW_DATA_PATH / "household_energy_consumption.csv")

    print(f"Before cleaning: {len(data_frame)} rows")

    data_frame.drop_duplicates()
    data_frame.dropna()

    data_frame["Date"] = pd.to_datetime(data_frame["Date"])
    data_frame["Month"] = data_frame["Date"].dt.month
    data_frame["DayOfWeek"] = data_frame["Date"].dt.dayofweek
    data_frame["Has_AC"] = (
        data_frame["Has_AC"].str.strip().str.lower().map({"no": 0, "yes": 1})
    )
    data_frame = data_frame.drop(columns=["Date", "Household_ID"])

    data_frame.to_csv(PROCESSED_DATA_PATH / "energy.csv", index=False)

    train, temp = train_test_split(data_frame, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_csv(SPLITS_DATA_PATH / "train.csv", index=False)
    val.to_csv(SPLITS_DATA_PATH / "val.csv", index=False)
    test.to_csv(SPLITS_DATA_PATH / "test.csv", index=False)

    print(f"Train: {(len(train))} Val: {(len(val))} Test {(len(test))}")


if __name__ == "__main__":
    main()
