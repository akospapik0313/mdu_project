from __future__ import annotations

import pandas as pd


def load_id_vwap_for_timestamps(path, target_timestamps_utc):
    if not path.exists():
        raise FileNotFoundError(
            f"ID price file not found:\n{path}"
        )

    df = pd.read_excel(path)

    required_columns = {"timestamp", "vwap"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "id_price.xlsx is missing columns: "
            f"{sorted(missing_columns)}"
        )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    series = (
        df[["timestamp_utc", "vwap"]]
        .dropna(subset=["vwap"])
        .groupby("timestamp_utc")["vwap"]
        .mean()
        .sort_index()
    )

    target_index = pd.DatetimeIndex(
        target_timestamps_utc
    )

    combined_index = (
        series.index
        .union(target_index)
        .sort_values()
    )

    aligned = (
        series
        .reindex(combined_index)
        .interpolate(method="time")
        .ffill()
        .bfill()
        .reindex(target_index)
    )

    if aligned.isna().any():
        raise ValueError(
            "ID VWAP could not be aligned to all schedule timestamps."
        )

    return aligned.to_numpy(dtype=float)