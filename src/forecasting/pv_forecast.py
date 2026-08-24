from src.forecasting.common import (
    DATA_DIR,
    build_xgb_model,
    get_train_and_forecast,
    load_params,
    make_forecast_matrix,
    make_training_matrix,
    parse_args,
    read_enriched_excel,
    run_hpo,
    save_forecast_excel,
    save_params,
)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = (
    DATA_DIR
    / "enriched"
    / "pv"
    / "pv_enriched.xlsx"
)

PARAM_PATH = (
    DATA_DIR
    / "parameters"
    / "hp_pv_fc.json"
)

OUTPUT_DIR = (
    DATA_DIR
    / "output_data"
    / "forecast"
    / "pv"
)


# ============================================================
# MODEL CONTRACT
# ============================================================

TARGET = "pv_target_kw"

FEATURES = [
    # ECMWF weather
    "wind_speed_10m",
    "temperature_2m",

    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",

    "shortwave_radiation",
    "diffuse_radiation",
    "direct_horizontal_radiation",
    "diffuse_fraction",

    # Solar geometry
    "solar_azimuth",
    "solar_elevation",
    "solar_zenith",

    "solar_azimuth_sin",
    "solar_azimuth_cos",
    "solar_elevation_sin",
    "solar_elevation_cos",

    "is_daylight",

    # Periodic time features
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",

    # Historical PV
    "pv_power_lag_48h",
    "pv_power_lag_168h",
]


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args(
        "Day-ahead PV forecast"
    )

    day = args.day

    print(
        "\n"
        "============================================\n"
        "PV FORECAST\n"
        "============================================"
    )

    print(
        f"\nDelivery day: {day}"
    )

    print(
        f"HPO: {args.hpo}"
    )

    df = read_enriched_excel(
        INPUT_PATH
    )

    train_df, forecast_df = (
        get_train_and_forecast(
            df=df,
            day=day,
        )
    )

    X_train, y_train = (
        make_training_matrix(
            train_df=train_df,
            features=FEATURES,
            target=TARGET,
        )
    )

    X_forecast = (
        make_forecast_matrix(
            forecast_df=forecast_df,
            features=FEATURES,
        )
    )

    # ========================================================
    # HPO OR SAVED PARAMETERS
    # ========================================================

    if args.hpo:

        params = run_hpo(
            X=X_train,
            y=y_train,
            split_mode="pv_blocked",
        )

        save_params(
            params=params,
            path=PARAM_PATH,
        )

    else:

        params = load_params(
            PARAM_PATH
        )

    # ========================================================
    # FINAL MODEL
    # ========================================================

    print(
        "\nTraining final XGBoost model..."
    )

    model = build_xgb_model(
        params
    )

    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(
        X_forecast
    )

    # PV cannot be negative.
    prediction = prediction.clip(
        min=0.0
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output = forecast_df[
        [
            "timestamp",
            "timestamp_utc",
        ]
    ].copy()

    output[
        "pv_forecast_kw"
    ] = prediction

    output_path = (
        OUTPUT_DIR
        / f"pv_forecast_{day}.xlsx"
    )

    save_forecast_excel(
        df=output,
        path=output_path,
    )

    print(
        "\nPV forecast complete."
    )


if __name__ == "__main__":
    main()
