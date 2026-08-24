from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# COLUMN HELPERS
# ============================================================

def _find_first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = False,
    label: str = "column",
) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise KeyError(
            f"Could not find required {label}. Tried: {list(candidates)}"
        )

    return None


def _get_timestamp_column(df: pd.DataFrame) -> str:
    return _find_first_existing_column(
        df,
        ["timestamp", "datetime", "date", "time"],
        required=True,
        label="timestamp column",
    )


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ts_col = _get_timestamp_column(df)
    df[ts_col] = pd.to_datetime(df[ts_col])

    df = df.sort_values(ts_col).reset_index(drop=True)
    return df


def load_schedule(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Schedule file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}"
        )

    return _ensure_datetime_index(df)


# ============================================================
# DATA EXTRACTION
# ============================================================

def _extract_series(df: pd.DataFrame) -> dict[str, pd.Series | None]:
    ts_col = _get_timestamp_column(df)

    load_col = _find_first_existing_column(
        df,
        [
            "load_forecast_kw",
            "portfolio_load_kw",
            "expected_load_kw",
            "load_kw",
            "load",
        ],
    )

    pv_col = _find_first_existing_column(
        df,
        [
            "pv_forecast_kw",
            "pv_power_kw",
            "pv_generation_kw",
            "pv_kw",
            "pv",
        ],
    )

    dam_price_col = _find_first_existing_column(
        df,
        [
            "dam_price_forecast_eur_mwh",
            "dam_price_eur_mwh",
            "dam_price",
            "price_eur_mwh",
        ],
    )

    charge_col = _find_first_existing_column(
        df,
        [
            "battery_charge_kw",
            "bess_charge_kw",
            "charge_kw",
        ],
    )

    discharge_col = _find_first_existing_column(
        df,
        [
            "battery_discharge_kw",
            "bess_discharge_kw",
            "discharge_kw",
        ],
    )

    soc_col = _find_first_existing_column(
        df,
        [
            "soc_pct",
            "battery_soc_pct",
            "bess_soc_pct",
        ],
    )

    grid_import_col = _find_first_existing_column(
        df,
        [
            "grid_import_kw",
            "import_kw",
        ],
    )

    grid_export_col = _find_first_existing_column(
        df,
        [
            "grid_export_kw",
            "export_kw",
        ],
    )

    net_grid_col = _find_first_existing_column(
        df,
        [
            "net_grid_kw",
            "resulting_grid_kw",
            "grid_net_kw",
            "grid_power_kw",
        ],
    )

    if net_grid_col is not None:
        net_grid = df[net_grid_col]
    elif grid_import_col is not None or grid_export_col is not None:
        net_grid = (
            (df[grid_import_col] if grid_import_col else 0.0)
            - (df[grid_export_col] if grid_export_col else 0.0)
        )
    else:
        net_grid = None

    return {
        "timestamp": df[ts_col],
        "load": df[load_col] if load_col else None,
        "pv": df[pv_col] if pv_col else None,
        "dam_price": df[dam_price_col] if dam_price_col else None,
        "charge": df[charge_col] if charge_col else None,
        "discharge": df[discharge_col] if discharge_col else None,
        "soc": df[soc_col] if soc_col else None,
        "net_grid": net_grid,
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_da_schedule(
    schedule_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    df = load_schedule(schedule_path)
    s = _extract_series(df)

    ts = s["timestamp"]

    if output_path is None:
        day_str = pd.to_datetime(ts.iloc[0]).strftime("%Y-%m-%d")
        output_path = Path(
            f"data/output_data/plots/da/da_plot_{day_str}.png"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(16, 18),
        sharex=True,
        constrained_layout=True,
    )

    day_str = pd.to_datetime(ts.iloc[0]).strftime("%Y-%m-%d")
    fig.suptitle(
        f"Day-Ahead Schedule - {day_str}",
        fontsize=16,
        fontweight="bold",
    )

    # 1) Forecasted load
    ax = axes[0]
    if s["load"] is not None:
        ax.plot(ts, s["load"], linewidth=1.8, label="Forecasted load")
        ax.legend(loc="upper right")
    ax.set_ylabel("kW")
    ax.set_title("Forecasted Load")
    ax.grid(True, alpha=0.3)

    # 2) Forecasted PV
    ax = axes[1]
    if s["pv"] is not None:
        ax.plot(ts, s["pv"], linewidth=1.8, label="Forecasted PV generation")
        ax.legend(loc="upper right")
    ax.set_ylabel("kW")
    ax.set_title("Forecasted PV Generation")
    ax.grid(True, alpha=0.3)

    # 3) DAM price
    ax = axes[2]
    if s["dam_price"] is not None:
        ax.plot(ts, s["dam_price"], linewidth=1.8, label="DAM price")
        ax.legend(loc="upper right")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Day-Ahead Market Price")
    ax.grid(True, alpha=0.3)

    # 4) Battery charge / discharge
    ax = axes[3]
    has_handles = False
    if s["charge"] is not None:
        ax.step(ts, s["charge"], where="mid", label="Battery charge")
        has_handles = True
    if s["discharge"] is not None:
        ax.step(ts, -s["discharge"], where="mid", label="Battery discharge")
        has_handles = True
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("kW")
    ax.set_title("Battery Charge / Discharge")
    if has_handles:
        ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 5) SOC
    ax = axes[4]
    if s["soc"] is not None:
        ax.plot(ts, s["soc"], linewidth=1.8, label="SOC")
        ax.legend(loc="upper right")
    ax.set_ylabel("%")
    ax.set_title("State of Charge")
    ax.grid(True, alpha=0.3)

    # 6) Resulting net grid
    ax = axes[5]
    if s["net_grid"] is not None:
        ax.plot(ts, s["net_grid"], linewidth=1.8, label="Net grid power")
        ax.axhline(0.0, linewidth=1.0)
        ax.legend(loc="upper right")
    ax.set_ylabel("kW")
    ax.set_title("Resulting Net Grid Power")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule-path",
        required=True,
        help="Path to the DA schedule file.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output PNG path.",
    )
    args = parser.parse_args()

    saved = plot_da_schedule(
        schedule_path=args.schedule_path,
        output_path=args.output_path,
    )

    print("\n============================================================")
    print("DAY-AHEAD VISUALIZATION")
    print("============================================================")
    print(f"\nPlot saved:\n{saved}")


if __name__ == "__main__":
    main()