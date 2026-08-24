from __future__ import annotations

import argparse
import subprocess
import sys
import time


# ============================================================
# CLI
# ============================================================

def str_to_bool(value):
    value = str(value).lower()

    if value in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return True

    if value in {
        "false",
        "0",
        "no",
        "n",
    }:
        return False

    raise argparse.ArgumentTypeError(
        "Expected true or false."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run complete day-ahead forecasting, "
            "BESS optimization and visualization pipeline."
        )
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
        help=(
            "Run Optuna HPO for forecasting models: "
            "true/false"
        ),
    )

    return parser.parse_args()


# ============================================================
# COMMAND RUNNER
# ============================================================

def run_step(
    title,
    command,
):
    print(
        "\n"
        + "=" * 60
    )

    print(
        title
    )

    print(
        "=" * 60
    )

    print(
        "\nCommand:"
    )

    print(
        " ".join(command)
    )

    start = time.time()

    result = subprocess.run(
        command,
        check=False,
    )

    elapsed = (
        time.time()
        - start
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"\nPipeline stopped.\n"
            f"Failed step: {title}\n"
            f"Return code: {result.returncode}"
        )

    print(
        f"\nCompleted in "
        f"{elapsed:.1f} seconds."
    )


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

    python = sys.executable

    pipeline_start = time.time()

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
    # 1. LOAD FORECAST
    # ========================================================

    run_step(
        title="1/5 LOAD FORECAST",
        command=[
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
    # 2. PV FORECAST
    # ========================================================

    run_step(
        title="2/5 PV FORECAST",
        command=[
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
    # 3. DAM PRICE FORECAST
    # ========================================================

    run_step(
        title="3/5 DAM PRICE FORECAST",
        command=[
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
    # 4. DAY-AHEAD BESS OPTIMIZATION
    # ========================================================

    run_step(
        title="4/5 DAY-AHEAD BESS OPTIMIZATION",
        command=[
            python,
            "-m",
            "src.optimization.da_optimizer",
            "--day",
            day,
        ],
    )

    # ========================================================
    # 5. VISUALIZATION
    # ========================================================

    run_step(
        title="5/5 DAY-AHEAD VISUALIZATION",
        command=[
            python,
            "-m",
            "src.visualization.plot_da",
            "--day",
            day,
        ],
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    total_elapsed = (
        time.time()
        - pipeline_start
    )

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
        f"\nDelivery day: {day}"
    )

    print(
        f"Total runtime: "
        f"{total_elapsed:.1f} seconds"
    )

    print(
        "\nGenerated outputs:"
    )

    print(
        f"  data/output_data/forecast/load/"
        f"load_forecast_{day}.xlsx"
    )

    print(
        f"  data/output_data/forecast/pv/"
        f"pv_forecast_{day}.xlsx"
    )

    print(
        f"  data/output_data/forecast/dam/"
        f"dam_forecast_{day}.xlsx"
    )

    print(
        f"  data/output_data/schedules/da/"
        f"da_schedule_{day}.xlsx"
    )

    print(
        f"  data/output_data/plots/da/"
        f"da_plot_{day}.png"
    )


if __name__ == "__main__":
    main()
