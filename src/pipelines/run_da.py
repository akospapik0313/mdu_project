from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENRICHMENT_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "features"
    / "enrichment.py"
)

SCHEDULE_DIR = (
    PROJECT_ROOT
    / "data"
    / "output_data"
    / "schedules"
    / "da"
)

PLOT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output_data"
    / "plots"
    / "da"
)


# ============================================================
# CLI
# ============================================================

def str_to_bool(value):
    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "y"}:
        return True

    if value in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(
        "Expected true or false."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run complete day-ahead pipeline."
    )

    parser.add_argument(
        "--day",
        required=True,
        type=str,
        help="Delivery day, e.g. 2026-08-01",
    )

    parser.add_argument(
        "--hpo",
        default=False,
        type=str_to_bool,
        help="Run Optuna HPO: true/false",
    )

    return parser.parse_args()


# ============================================================
# PIPELINE STEP
# ============================================================

def run_step(title, command):
    print(
        "\n"
        + "=" * 60
    )

    print(title)

    print(
        "=" * 60
    )

    print(
        "\n"
        + " ".join(
            str(x)
            for x in command
        )
    )

    start = time.time()

    subprocess.run(
        [
            str(x)
            for x in command
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    elapsed = (
        time.time()
        - start
    )

    print(
        f"\nCompleted in {elapsed:.1f} s"
    )


# ============================================================
# VISUALIZATION
# ============================================================

def setup_time_axis(ax):
    ax.xaxis.set_major_locator(
        mdates.HourLocator(
            interval=2
        )
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%H:%M",
            tz="Europe/Stockholm",
        )
    )

    ax.grid(
        True,
        alpha=0.25,
    )


def create_da_plot(day):
    schedule_path = (
        SCHEDULE_DIR
        / f"da_schedule_{day}.xlsx"
    )

    if not schedule_path.exists():
        raise FileNotFoundError(
            f"DA schedule not found:\n"
            f"{schedule_path}"
        )

    df = pd.read_excel(
        schedule_path,
        sheet_name="schedule",
    )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
    )

    df["timestamp"] = (
        df["timestamp_utc"]
        .dt.tz_convert(
            "Europe/Stockholm"
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

    consumption_kw = pd.to_numeric(
        df["load_forecast_kw"],
        errors="coerce",
    )

    pv_kw = pd.to_numeric(
        df["pv_forecast_kw"],
        errors="coerce",
    )

    dam_price = pd.to_numeric(
        df["dam_price_forecast"],
        errors="coerce",
    )

    charge_kw = pd.to_numeric(
        df["bess_charge_kw"],
        errors="coerce",
    )

    discharge_kw = pd.to_numeric(
        df["bess_discharge_kw"],
        errors="coerce",
    )

    soc_percent = pd.to_numeric(
        df["bess_soc_percent"],
        errors="coerce",
    )

    grid_net_kw = pd.to_numeric(
        df["grid_net_kw"],
        errors="coerce",
    )

    net_before_bess_kw = (
        consumption_kw
        -
        pv_kw
    )

    ts = df["timestamp"]

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(16, 16),
        sharex=True,
    )

    ax1, ax2, ax3, ax4, ax5, ax6 = axes

    ax1.step(
        ts,
        consumption_kw,
        where="post",
        linewidth=2.0,
        label="Forecast load",
    )
    ax1.set_ylabel("kW")
    ax1.set_title("Forecasted Load")
    ax1.legend(loc="upper left")
    setup_time_axis(ax1)

    ax2.step(
        ts,
        pv_kw,
        where="post",
        linewidth=2.0,
        label="Forecast PV",
    )
    ax2.set_ylabel("kW")
    ax2.set_title("Forecasted PV Generation")
    ax2.legend(loc="upper left")
    setup_time_axis(ax2)

    ax3.step(
        ts,
        dam_price,
        where="post",
        linewidth=2.0,
        label="DAM price",
    )
    ax3.set_ylabel("EUR/MWh")
    ax3.set_title("Forecasted Day-Ahead Market Price")
    ax3.legend(loc="upper left")
    setup_time_axis(ax3)

    ax4.step(
        ts,
        charge_kw,
        where="post",
        linewidth=2.0,
        label="Charge",
    )
    ax4.step(
        ts,
        -discharge_kw,
        where="post",
        linewidth=2.0,
        label="Discharge",
    )
    ax4.axhline(
        0,
        linewidth=0.8,
    )
    ax4.set_ylabel("kW")
    ax4.set_title("BESS Charge / Discharge")
    ax4.legend(loc="upper left")
    setup_time_axis(ax4)

    ax5.step(
        ts,
        soc_percent,
        where="post",
        linewidth=2.0,
        label="SOC",
    )
    ax5.set_ylabel("%")
    ax5.set_ylim(0, 100)
    ax5.set_title("State of Charge (SOC)")
    ax5.legend(loc="upper left")
    setup_time_axis(ax5)

    ax6.step(
        ts,
        net_before_bess_kw,
        where="post",
        linewidth=2.0,
        linestyle="--",
        label="Net load before BESS",
    )
    ax6.step(
        ts,
        grid_net_kw,
        where="post",
        linewidth=2.2,
        label="Resulting grid load",
    )
    ax6.axhline(
        0,
        linewidth=0.8,
    )
    ax6.set_ylabel("kW")
    ax6.set_title("Resulting Net Grid Power")
    ax6.legend(loc="upper left")
    setup_time_axis(ax6)
    ax6.set_xlabel(
        "Local Time (Europe/Stockholm)"
    )

    fig.suptitle(
        f"MDU Energy Community - Day-Ahead Schedule - {day}",
        fontsize=16,
        fontweight="bold",
    )

    # No subtitle or energy-summary line is shown.
    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PLOT_DIR
        / f"da_plot_{day}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path
# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    day = args.day

    hpo = (
        "true"
        if args.hpo
        else "false"
    )

    python = (
        sys.executable
    )

    pipeline_start = (
        time.time()
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MDU ENERGY COMMUNITY - DAY-AHEAD PIPELINE"
    )

    print(
        "=" * 60
    )

    print(
        f"\nDelivery day: {day}"
    )

    print(
        f"HPO:          {hpo}"
    )

    # ========================================================
    # 1. ENRICHMENT
    # ========================================================

    if not ENRICHMENT_SCRIPT.exists():
        raise FileNotFoundError(
            "Enrichment script not found:\n"
            f"{ENRICHMENT_SCRIPT}"
        )

    run_step(
        "1/6 DATA ENRICHMENT",
        [
            python,
            ENRICHMENT_SCRIPT,
        ],
    )

    # ========================================================
    # 2. LOAD FORECAST
    # ========================================================

    run_step(
        "2/6 LOAD FORECAST",
        [
            python,
            "-m",
            "src.forecasting.load_forecast",
            "--day",
            day,
            "--hpo",
            hpo,
        ],
    )

    # ========================================================
    # 3. PV FORECAST
    # ========================================================

    run_step(
        "3/6 PV FORECAST",
        [
            python,
            "-m",
            "src.forecasting.pv_forecast",
            "--day",
            day,
            "--hpo",
            hpo,
        ],
    )

    # ========================================================
    # 4. DAM FORECAST
    # ========================================================

    run_step(
        "4/6 DAM PRICE FORECAST",
        [
            python,
            "-m",
            "src.forecasting.dam_forecast",
            "--day",
            day,
            "--hpo",
            hpo,
        ],
    )

    # ========================================================
    # 5. OPTIMIZATION
    # ========================================================

    run_step(
        "5/6 DAY-AHEAD BESS OPTIMIZATION",
        [
            python,
            "-m",
            "src.optimization.da_optimizer",
            "--day",
            day,
        ],
    )

    # ========================================================
    # 6. VISUALIZATION
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "6/6 DAY-AHEAD VISUALIZATION"
    )

    print(
        "=" * 60
    )

    plot_start = (
        time.time()
    )

    plot_path = (
        create_da_plot(
            day
        )
    )

    print(
        f"\nVisualization saved:\n"
        f"{plot_path}"
    )

    print(
        f"\nCompleted in "
        f"{time.time() - plot_start:.1f} s"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "DAY-AHEAD PIPELINE COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"\nTotal runtime: "
        f"{time.time() - pipeline_start:.1f} s"
    )


if __name__ == "__main__":
    main()
