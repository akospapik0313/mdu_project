from pathlib import Path

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
    / "load"
    / "load_enriched.xlsx"
)

PARAM_PATH = (
    DATA_DIR
    / "parameters"
    / "hp_load_fc.json"
)

OUTPUT_DIR = (
    DATA_DIR
    / "output_data"
    / "forecast"
    / "load"
)


# ============================================================
# MODEL CONTRACT
# ============================================================

TARGET = "load_target_kw"

FEATURES = [
    "load_lag_48h",
    "load_lag_168h",

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
        "Day-ahead community load forecast"
    )

    day = args.day

    print(
        "\n"
        "============================================\n"
        "LOAD FORECAST\n"
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
    #
    # Train on the entire usable previous-year dataset.
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

    # Physical load cannot be negative.
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
        "load_forecast_kw"
    ] = prediction

    output_path = (
        OUTPUT_DIR
        / f"load_forecast_{day}.xlsx"
    )

    save_forecast_excel(
        df=output,
        path=output_path,
    )

    print(
        "\nLOAD forecast complete."
    )


if __name__ == "__main__":
    main()
