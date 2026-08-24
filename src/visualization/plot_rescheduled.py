from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLOT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "output_data"
    / "plots"
    / "rescheduled"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--schedule-file",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--day",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--as-of",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "DA",
            "ID",
        ],
    )

    return parser.parse_args()


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


def create_rescheduled_plot(
    schedule_file,
    day,
    as_of,
    stage,
):
    schedule_path = Path(
        schedule_file
    )

    if not schedule_path.exists():

        raise FileNotFoundError(
            f"Rescheduled schedule not found:\n"
            f"{schedule_path}"
        )

    df = pd.read_excel(
        schedule_path,
        sheet_name="schedule",
    )

    summary_df = pd.read_excel(
        schedule_path,
        sheet_name="summary",
    )

    df[
        "timestamp_utc"
    ] = pd.to_datetime(
        df[
            "timestamp_utc"
        ],
        utc=True,
    )

    df[
        "timestamp"
    ] = (
        df[
            "timestamp_utc"
        ]
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

    summary = dict(
        zip(
            summary_df[
                "metric"
            ],
            summary_df[
                "value"
            ],
        )
    )

    if (
        "effective_load_kw"
        in df.columns
    ):

        consumption_kw = df[
            "effective_load_kw"
        ]

    else:

        consumption_kw = df[
            "load_forecast_kw"
        ]

    net_before_bess_kw = (
        consumption_kw
        -
        df[
            "pv_forecast_kw"
        ]
    )

    ts = df[
        "timestamp"
    ]

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(
            16,
            16,
        ),
        sharex=True,
    )

    ax1, ax2, ax3, ax4, ax5, ax6 = axes

    # 1. Expected consumption
    ax1.step(
        ts,
        consumption_kw,
        where="post",
        linewidth=2.0,
        label="Expected consumption",
    )

    ax1.set_ylabel(
        "kW"
    )

    ax1.set_title(
        "Várható fogyasztás"
    )

    ax1.legend(
        loc="upper left"
    )

    setup_time_axis(
        ax1
    )

    # 2. Expected PV
    ax2.step(
        ts,
        df[
            "pv_forecast_kw"
        ],
        where="post",
        linewidth=2.0,
        label="Expected PV",
    )

    ax2.set_ylabel(
        "kW"
    )

    ax2.set_title(
        "Várható PV termelés"
    )

    ax2.legend(
        loc="upper left"
    )

    setup_time_axis(
        ax2
    )

    # 3. DAM price
    ax3.step(
        ts,
        df[
            "dam_price_forecast"
        ],
        where="post",
        linewidth=2.0,
        label="DAM price",
    )

    ax3.set_ylabel(
        "EUR/MWh"
    )

    ax3.set_title(
        "DAM ár"
    )

    ax3.legend(
        loc="upper left"
    )

    setup_time_axis(
        ax3
    )

    # 4. BESS
    ax4.step(
        ts,
        df[
            "bess_charge_kw"
        ],
        where="post",
        linewidth=2.0,
        label="Charge",
    )

    ax4.step(
        ts,
        -df[
            "bess_discharge_kw"
        ],
        where="post",
        linewidth=2.0,
        label="Discharge",
    )

    ax4.axhline(
        0,
        linewidth=0.8,
    )

    ax4.set_ylabel(
        "kW"
    )

    ax4.set_title(
        "Akksi töltése / kisütése"
    )

    ax4.legend(
        loc="upper left"
    )

    setup_time_axis(
        ax4
    )

    # 5. SOC
    ax5.step(
        ts,
        df[
            "bess_soc_percent"
        ],
        where="post",
        linewidth=2.0,
        label="SOC",
    )

    ax5.set_ylabel(
        "%"
    )

    ax5.set_ylim(
        0,
        100,
    )

    ax5.set_title(
        "State of Charge (SOC)"
    )

    ax5.legend(
        loc="upper left"
    )

    setup_time_axis(
        ax5
    )

    # 6. Resulting consumption curve
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
        df[
            "grid_net_kw"
        ],
        where="post",
        linewidth=2.2,
        label="Resulting grid load",
    )

    ax6.axhline(
        0,
        linewidth=0.8,
    )

    ax6.set_ylabel(
        "kW"
    )

    ax6.set_title(
        "Eredő fogyasztási görbe"
    )

    ax6.legend(
        loc="upper left"
    )

    setup_time_axis(
        ax6
    )

    ax6.set_xlabel(
        "Local time (Europe/Stockholm)"
    )

    requested = float(
        summary.get(
            "company_ev_requested_kwh",
            0.0,
        )
    )

    delivered = float(
        summary.get(
            "company_ev_delivered_kwh",
            0.0,
        )
    )

    unserved = float(
        summary.get(
            "company_ev_unserved_kwh",
            0.0,
        )
    )

    if stage == "DA":

        market_cost_text = (
            f"DAM cost: "
            f"{float(summary.get('dam_energy_cost_eur', 0.0)):.2f} EUR"
        )

        stage_text = (
            "PRE-GATE DAY-AHEAD RESCHEDULING"
        )

    else:

        market_cost_text = (
            f"ID trade cost: "
            f"{float(summary.get('id_trade_cost_eur', 0.0)):.2f} EUR"
        )

        stage_text = (
            "POST-GATE INTRADAY RESCHEDULING"
        )

    summary_text = (
        f"{stage_text}"
        f"   |   As-of: {as_of}"
        f"   |   EV request: {requested / 1000:.2f} MWh"
        f"   |   EV served: {delivered / 1000:.2f} MWh"
        f"   |   EV unserved: {unserved / 1000:.2f} MWh"
        f"   |   {market_cost_text}"
    )

    fig.suptitle(
        (
            "MDU Energy Community - "
            f"{stage} Rescheduled Operation - {day}"
        ),
        fontsize=16,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.965,
        summary_text,
        ha="center",
        fontsize=9.5,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.955,
        ]
    )

    stage_dir = (
        PLOT_ROOT
        / stage.lower()
    )

    stage_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # One CURRENT plot per delivery day and market stage.
    # Every new rescheduling run overwrites the previous plot.
    output_path = (
        stage_dir
        / (
            f"{stage.lower()}_rescheduled_plot_"
            f"{day}.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


def main():
    args = parse_args()

    output_path = create_rescheduled_plot(
        schedule_file=args.schedule_file,
        day=args.day,
        as_of=args.as_of,
        stage=args.stage,
    )

    print(
        f"\nRescheduled visualization saved:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
