from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


# ============================================================
# GLOBAL SETTINGS
# ============================================================

TIMEZONE = "Europe/Stockholm"
RANDOM_STATE = 42
N_SPLITS = 4
N_TRIALS = 100


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)


# ============================================================
# CLI HELPERS
# ============================================================

def str_to_bool(value):
    value = str(value).lower()

    if value in {"true", "1", "yes", "y"}:
        return True

    if value in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(
        "Expected true or false."
    )


def parse_args(description):
    parser = argparse.ArgumentParser(
        description=description
    )

    parser.add_argument(
        "--day",
        required=True,
        type=str,
        help="Delivery day, e.g. 2026-07-07",
    )

    parser.add_argument(
        "--hpo",
        default=False,
        type=str_to_bool,
        help="Run 100-trial Optuna HPO: true/false",
    )

    return parser.parse_args()


# ============================================================
# DATA HELPERS
# ============================================================

def read_enriched_excel(path):
    df = pd.read_excel(
        path
    )

    # All enriched Excel timestamps were saved as strings.
    # Parse through UTC and reconstruct Stockholm local time.
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
    )

    df["timestamp"] = (
        df["timestamp_utc"]
        .dt.tz_convert(
            TIMEZONE
        )
    )

    df = (
        df
        .sort_values(
            "timestamp_utc"
        )
        .reset_index(
            drop=True
        )
    )

    return df


def get_train_and_forecast(
    df,
    day,
):
    """
    Forecast day is interpreted as a Stockholm LOCAL date.

    Training window:
        exactly the previous calendar year

    Example:
        forecast day: 2026-07-07
        training:     2025-07-07 00:00 local
                      up to 2026-07-07 00:00 local
    """

    forecast_start = (
        pd.Timestamp(day)
        .tz_localize(
            TIMEZONE
        )
    )

    forecast_end = (
        forecast_start
        + pd.DateOffset(
            days=1
        )
    )

    train_start = (
        forecast_start
        - pd.DateOffset(
            years=1
        )
    )

    train_df = (
        df.loc[
            (
                df["timestamp"]
                >= train_start
            )
            &
            (
                df["timestamp"]
                < forecast_start
            )
        ]
        .copy()
    )

    forecast_df = (
        df.loc[
            (
                df["timestamp"]
                >= forecast_start
            )
            &
            (
                df["timestamp"]
                < forecast_end
            )
        ]
        .copy()
    )

    if train_df.empty:
        raise ValueError(
            "Training window is empty."
        )

    if forecast_df.empty:
        raise ValueError(
            f"No rows found for forecast day {day}."
        )

    print(
        "\nTraining window:"
    )

    print(
        f"  {train_df['timestamp'].min()}"
    )

    print(
        "  ->"
    )

    print(
        f"  {train_df['timestamp'].max()}"
    )

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        "\nForecast window:"
    )

    print(
        f"  {forecast_df['timestamp'].min()}"
    )

    print(
        "  ->"
    )

    print(
        f"  {forecast_df['timestamp'].max()}"
    )

    print(
        f"Forecast rows: {len(forecast_df):,}"
    )

    return (
        train_df,
        forecast_df,
    )


# ============================================================
# HPO PARAMETER FILES
# ============================================================

def save_params(
    params,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            params,
            file,
            indent=4,
        )

    print(
        f"\nBest parameters saved to:"
    )

    print(
        f"  {path}"
    )


def load_params(path):
    if not path.exists():
        raise FileNotFoundError(
            f"HPO parameter file does not exist:\n"
            f"{path}\n\n"
            f"Run the model once with --hpo true."
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        params = json.load(
            file
        )

    print(
        "\nLoaded parameters:"
    )

    print(
        f"  {path}"
    )

    return params


# ============================================================
# XGBOOST
# ============================================================

def build_xgb_model(params):
    """
    Common point-forecast XGBoost regressor.
    """

    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
        **params,
    )


def suggest_xgb_params(trial):
    """
    Shared HPO search space for all three models.
    """

    return {
        "n_estimators":
            trial.suggest_int(
                "n_estimators",
                200,
                1200,
                step=100,
            ),

        "max_depth":
            trial.suggest_int(
                "max_depth",
                3,
                12,
            ),

        "learning_rate":
            trial.suggest_float(
                "learning_rate",
                0.01,
                0.2,
                log=True,
            ),

        "min_child_weight":
            trial.suggest_float(
                "min_child_weight",
                1.0,
                20.0,
            ),

        "subsample":
            trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),

        "colsample_bytree":
            trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),

        "reg_alpha":
            trial.suggest_float(
                "reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),

        "reg_lambda":
            trial.suggest_float(
                "reg_lambda",
                1e-3,
                20.0,
                log=True,
            ),
    }


# ============================================================
# TEMPORAL CV
# ============================================================

def expanding_window_splits(
    n_samples,
):
    """
    LOAD and DAM.

    Standard 4-fold expanding-window TimeSeriesSplit.

    Training always occurs before validation.
    No shuffle.
    """

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    return list(
        splitter.split(
            np.arange(
                n_samples
            )
        )
    )


def pv_temporal_splits(
    n_samples,
):
    """
    PV.

    Four chronological, non-random, non-expanding folds.

    The data is divided into five consecutive blocks:

        B1 B2 B3 B4 B5

    Fold 1: train B1 -> validate B2
    Fold 2: train B2 -> validate B3
    Fold 3: train B3 -> validate B4
    Fold 4: train B4 -> validate B5

    This behaves like blocked temporal K-fold:
    - no shuffle
    - no future observations in training
    - training window does not expand
    """

    indices = np.arange(
        n_samples
    )

    blocks = np.array_split(
        indices,
        N_SPLITS + 1,
    )

    splits = []

    for fold in range(
        N_SPLITS
    ):
        train_idx = blocks[
            fold
        ]

        valid_idx = blocks[
            fold + 1
        ]

        splits.append(
            (
                train_idx,
                valid_idx,
            )
        )

    return splits


# ============================================================
# OPTUNA
# ============================================================

def run_hpo(
    X,
    y,
    split_mode,
):
    """
    100-trial Optuna optimization.

    Objective:
        mean validation MAE across 4 temporal folds.
    """

    if split_mode == "expanding":
        splits = expanding_window_splits(
            len(X)
        )

    elif split_mode == "pv_blocked":
        splits = pv_temporal_splits(
            len(X)
        )

    else:
        raise ValueError(
            f"Unknown split mode: {split_mode}"
        )

    optuna.logging.set_verbosity(
        optuna.logging.WARNING
    )

    def objective(trial):
        params = suggest_xgb_params(
            trial
        )

        fold_mae = []

        for (
            train_idx,
            valid_idx,
        ) in splits:

            X_train = X.iloc[
                train_idx
            ]

            y_train = y.iloc[
                train_idx
            ]

            X_valid = X.iloc[
                valid_idx
            ]

            y_valid = y.iloc[
                valid_idx
            ]

            model = build_xgb_model(
                params
            )

            model.fit(
                X_train,
                y_train,
            )

            prediction = model.predict(
                X_valid
            )

            mae = mean_absolute_error(
                y_valid,
                prediction,
            )

            fold_mae.append(
                mae
            )

        return float(
            np.mean(
                fold_mae
            )
        )

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    print(
        f"\nStarting Optuna HPO: {N_TRIALS} trials..."
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=1,
        show_progress_bar=True,
    )

    print(
        "\nHPO complete."
    )

    print(
        f"Best CV MAE: "
        f"{study.best_value:.6f}"
    )

    print(
        "\nBest parameters:"
    )

    for key, value in (
        study.best_params.items()
    ):
        print(
            f"  {key}: {value}"
        )

    return study.best_params


# ============================================================
# TRAINING MATRIX
# ============================================================

def make_training_matrix(
    train_df,
    features,
    target,
):
    """
    Drop only rows that cannot be used by the selected model.
    This naturally removes leading lag/rolling NaNs.
    """

    model_df = (
        train_df[
            features
            + [target]
        ]
        .dropna()
        .copy()
    )

    if model_df.empty:
        raise ValueError(
            "No usable training rows remain "
            "after removing missing model features."
        )

    X = (
        model_df[
            features
        ]
    )

    y = (
        model_df[
            target
        ]
    )

    print(
        f"\nUsable model-training rows: "
        f"{len(model_df):,}"
    )

    return (
        X,
        y,
    )


def make_forecast_matrix(
    forecast_df,
    features,
):
    """
    Forecast-day features must be present.
    """

    missing = (
        forecast_df[
            features
        ]
        .isna()
        .sum()
    )

    missing = (
        missing[
            missing > 0
        ]
    )

    if not missing.empty:
        raise ValueError(
            "Forecast day contains missing model features:\n"
            f"{missing}"
        )

    return forecast_df[
        features
    ]


# ============================================================
# OUTPUT
# ============================================================

def save_forecast_excel(
    df,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = df.copy()

    for column in [
        "timestamp",
        "timestamp_utc",
    ]:
        if column in out.columns:
            out[column] = (
                out[column]
                .astype(str)
            )

    out.to_excel(
        path,
        index=False,
    )

    print(
        "\nForecast saved:"
    )

    print(
        f"  {path}"
    )
