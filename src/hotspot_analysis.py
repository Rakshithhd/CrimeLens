import pandas as pd


DATA_PATH = "data/raw/Train.xlsx - Train.csv"


def load_incident_data():
    """Load incident records and remove invalid geographical coordinates."""
    columns = [
        "TYPE", "NEIGHBOURHOOD", "Latitude", "Longitude",
        "YEAR", "MONTH"
    ]

    df = pd.read_csv(DATA_PATH, usecols=columns)

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")

    # Keep valid Vancouver-area coordinates; remove zero/missing locations.
    df = df.dropna(
        subset=["TYPE", "NEIGHBOURHOOD", "Latitude", "Longitude", "YEAR", "MONTH"]
    )

    df = df[
        df["Latitude"].between(49.0, 50.0)
        & df["Longitude"].between(-125.0, -122.5)
    ].copy()

    df["YEAR"] = df["YEAR"].astype(int)
    df["MONTH"] = df["MONTH"].astype(int)

    return df


def aggregate_hotspots(df):
    """Aggregate nearby incidents into weighted map points."""
    return (
        df.assign(
            Latitude=df["Latitude"].round(4),
            Longitude=df["Longitude"].round(4),
        )
        .groupby(["Latitude", "Longitude"], as_index=False)
        .size()
        .rename(columns={"size": "Incident_Count"})
    )
