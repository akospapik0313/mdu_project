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
    / "dam"
    / "dam_enriched.xlsx"
)

PARAM_PATH = (
    DATA_DIR
    / "parameters"
    / "hp_dam_fc.json"
)

OUTPUT_DIR = (
    DATA_DIR
    / "output_data"
    / "forecast"
    / "dam"
)


# ============================================================
# MODEL CONTRACT
# ============================================================

TARGET = "dam_price"

FEATURES = [
    # Day-ahead fundamentals
    "load_fc",
    "wind_onshore_fc",
    "solar_fc",
    "residual_load",

    # Price history
    "dam_price_lag_24h",
    "dam_price_lag_168h",

    # 24h lag rolling statistics
    "dam_price_lag_24h_mean_2h",
    "dam_price_lag_24h_std_2h",
    "dam_price_lag_24h_mean_8h",
    "dam_price_lag_24h_std_8h",
    "dam_price_lag_24h_mean_24h",
    "dam_price_lag_24h_std_24h",

    # Weekly lag rolling statistics
    "dam_price_lag_168h_mean_2h",
    "dam_price_lag_168h_std_2h",
    "dam_price_lag_168h_mean_8h",
    "dam_price_lag_168h_std_8h",
    "dam_price_lag_168h_mean_24h",
    "dam_price_lag_168h_std_24h",

    # Calendar
    "is_weekday",
    "is_weekend",
    "is_holiday",

    "hour_sin",
    "hour_cos",

    "doy_sin",
    "doy_cos",
]


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args(
        "Day-ahead electricity price forecast"
    )

    day = args.day

    print(
        "\n"
        "============================================\n"
        "DAM PRICE FORECAST\n"
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
            split_mode="expanding",
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

    # IMPORTANT:
    # DAM prices are NOT clipped.
    # Negative electricity prices are possible.

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
        "dam_price_forecast"
    ] = prediction

    output_path = (
        OUTPUT_DIR
        / f"dam_forecast_{day}.xlsx"
    )

    save_forecast_excel(
        df=output,
        path=output_path,
    )

    print(
        "\nDAM forecast complete."
    )


if __name__ == "__main__":
    main()
