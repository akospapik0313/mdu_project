from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import pvlib


# ============================================================
# SETTINGS
# ============================================================

TIMEZONE = "Europe/Stockholm"

LATITUDE = 59.3294
LONGITUDE = 18.0687

STEPS_PER_HOUR = 4

LAG_24H = 24 * STEPS_PER_HOUR
LAG_48H = 48 * STEPS_PER_HOUR
LAG_168H = 168 * STEPS_PER_HOUR

ROLLING_WINDOWS = {
    "2h": 2 * STEPS_PER_HOUR,
    "8h": 8 * STEPS_PER_HOUR,
    "24h": 24 * STEPS_PER_HOUR,
}


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

LOAD_INPUT = DATA_DIR / "input_data" / "load" / "load.xlsx"
PV_INPUT = DATA_DIR / "input_data" / "pv" / "pv_input.xlsx"
DAM_INPUT = DATA_DIR / "input_data" / "dam_price" / "dam_price.xlsx"

LOAD_OUTPUT = DATA_DIR / "enriched" / "load" / "load_enriched.xlsx"
PV_OUTPUT = DATA_DIR / "enriched" / "pv" / "pv_enriched.xlsx"
DAM_OUTPUT = DATA_DIR / "enriched" / "dam" / "dam_enriched.xlsx"


# ============================================================
# BASIC HELPERS
# ============================================================

def prepare_timestamp(df):
    """
    Input timestamp example:
        2025-01-01 00:00:00+01:00

    Internal representation:
        timestamp     -> Europe/Stockholm
        timestamp_utc -> UTC
    """

    df = df.copy()

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df["timestamp"] = (
        df["timestamp_utc"]
        .dt.tz_convert(TIMEZONE)
    )

    df = (
        df
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    return df


def add_calendar_features(df):
    """
    Common time/calendar features.
    All calendar information is based on Stockholm local time.
    """

    df = df.copy()

    ts = df["timestamp"]

    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["weekday"] = ts.dt.weekday
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear

    df["is_weekday"] = (
        df["weekday"] < 5
    ).astype(int)

    df["is_weekend"] = (
        df["weekday"] >= 5
    ).astype(int)

    years = sorted(
        ts.dt.year.unique()
    )

    swedish_holidays = holidays.country_holidays(
        "SE",
        years=years,
    )

    df["is_holiday"] = (
        ts.dt.date
        .map(
            lambda d: int(
                d in swedish_holidays
            )
        )
    )

    # Cyclic time features
    fractional_hour = (
        df["hour"]
        + df["minute"] / 60
    )

    df["hour_sin"] = np.sin(
        2 * np.pi * fractional_hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * fractional_hour / 24
    )

    df["doy_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    df["doy_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    return df


def save_excel(df, path):
    """
    Excel cannot store timezone-aware datetime values.
    Only the exported copy is converted to strings.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = df.copy()

    out["timestamp"] = (
        out["timestamp"]
        .astype(str)
    )

    out["timestamp_utc"] = (
        out["timestamp_utc"]
        .astype(str)
    )

    out.to_excel(
        path,
        index=False,
    )


# ============================================================
# LOAD ENRICHMENT
# ============================================================

def enrich_load():
    print("Creating LOAD enriched dataset...")

    df = pd.read_excel(
        LOAD_INPUT
    )

    df = prepare_timestamp(
        df
    )

    # --------------------------------------------------------
    # Forecast target
    #
    # Ad-hoc logistics_event_kw is intentionally excluded.
    # --------------------------------------------------------

    df["load_target_kw"] = (
        df["logistics_baseload_kw"]
        + df["manufacturing_load_kw"]
        + df["ev_workplace_kw"]
        + df["ev_fleet_kw"]
        + df["office_kw"]
    )

    # Physical total load including historical logistics events
    df["load_actual_kw"] = (
        df["load_target_kw"]
        + df["logistics_event_kw"]
    )

    # --------------------------------------------------------
    # Historical lags
    # --------------------------------------------------------

    df["load_lag_48h"] = (
        df["load_target_kw"]
        .shift(LAG_48H)
    )

    df["load_lag_168h"] = (
        df["load_target_kw"]
        .shift(LAG_168H)
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    df = add_calendar_features(
        df
    )

    save_excel(
        df,
        LOAD_OUTPUT,
    )

    print(
        f"LOAD saved: {LOAD_OUTPUT}"
    )


# ============================================================
# PV ENRICHMENT
# ============================================================

def enrich_pv():
    print("Creating PV enriched dataset...")

    df = pd.read_excel(
        PV_INPUT
    )

    df = prepare_timestamp(
        df
    )

    # --------------------------------------------------------
    # Target
    #
    # NaN values are allowed in power because this is the
    # target variable.
    # --------------------------------------------------------

    df["pv_target_kw"] = (
        df["power"]
    )

    # --------------------------------------------------------
    # Radiation features
    # --------------------------------------------------------

    df["direct_horizontal_radiation"] = (
        df["shortwave_radiation"]
        - df["diffuse_radiation"]
    ).clip(lower=0)

    df["diffuse_fraction"] = np.where(
        df["shortwave_radiation"] > 0,
        (
            df["diffuse_radiation"]
            / df["shortwave_radiation"]
        ),
        0,
    )

    df["diffuse_fraction"] = (
        df["diffuse_fraction"]
        .clip(0, 1)
    )

    # --------------------------------------------------------
    # Solar position
    # --------------------------------------------------------

    solar_position = (
        pvlib.solarposition.get_solarposition(
            time=pd.DatetimeIndex(
                df["timestamp_utc"]
            ),
            latitude=LATITUDE,
            longitude=LONGITUDE,
        )
    )

    df["solar_azimuth"] = (
        solar_position["azimuth"]
        .to_numpy()
    )

    df["solar_elevation"] = (
        solar_position["elevation"]
        .to_numpy()
    )

    df["solar_zenith"] = (
        solar_position["zenith"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Solar cyclic features
    # --------------------------------------------------------

    azimuth_rad = np.deg2rad(
        df["solar_azimuth"]
    )

    elevation_rad = np.deg2rad(
        df["solar_elevation"]
    )

    df["solar_azimuth_sin"] = np.sin(
        azimuth_rad
    )

    df["solar_azimuth_cos"] = np.cos(
        azimuth_rad
    )

    df["solar_elevation_sin"] = np.sin(
        elevation_rad
    )

    df["solar_elevation_cos"] = np.cos(
        elevation_rad
    )

    df["is_daylight"] = (
        df["solar_elevation"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Calendar / cyclic time features
    # --------------------------------------------------------

    df = add_calendar_features(
        df
    )

    # --------------------------------------------------------
    # Historical PV lags
    # --------------------------------------------------------

    df["pv_power_lag_48h"] = (
        df["power"]
        .shift(LAG_48H)
    )

    df["pv_power_lag_168h"] = (
        df["power"]
        .shift(LAG_168H)
    )

    save_excel(
        df,
        PV_OUTPUT,
    )

    print(
        f"PV saved: {PV_OUTPUT}"
    )


# ============================================================
# DAM ENRICHMENT
# ============================================================

def enrich_dam():
    print("Creating DAM enriched dataset...")

    df = pd.read_excel(
        DAM_INPUT
    )

    df = prepare_timestamp(
        df
    )

    # --------------------------------------------------------
    # Residual load
    # --------------------------------------------------------

    df["residual_load"] = (
        df["load_fc"]
        - df["wind_onshore_fc"]
        - df["solar_fc"]
    )

    # --------------------------------------------------------
    # Price lags
    # --------------------------------------------------------

    df["dam_price_lag_24h"] = (
        df["dam_price"]
        .shift(LAG_24H)
    )

    df["dam_price_lag_168h"] = (
        df["dam_price"]
        .shift(LAG_168H)
    )

    # --------------------------------------------------------
    # Rolling mean / std from the already lagged prices
    #
    # Windows:
    # 2h, 8h, 24h
    # --------------------------------------------------------

    for lag_name in [
        "24h",
        "168h",
    ]:

        lag_col = (
            f"dam_price_lag_{lag_name}"
        )

        for window_name, window_size in (
            ROLLING_WINDOWS.items()
        ):

            df[
                f"{lag_col}_mean_{window_name}"
            ] = (
                df[lag_col]
                .rolling(window_size)
                .mean()
            )

            df[
                f"{lag_col}_std_{window_name}"
            ] = (
                df[lag_col]
                .rolling(window_size)
                .std()
            )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    df = add_calendar_features(
        df
    )

    save_excel(
        df,
        DAM_OUTPUT,
    )

    print(
        f"DAM saved: {DAM_OUTPUT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "\n"
        "============================================\n"
        "MDU ENERGY COMMUNITY - DATA ENRICHMENT\n"
        "============================================\n"
    )

    enrich_load()
    enrich_pv()
    enrich_dam()

    print(
        "\n"
        "============================================\n"
        "ENRICHMENT COMPLETE\n"
        "============================================"
    )


if __name__ == "__main__":
    main()
