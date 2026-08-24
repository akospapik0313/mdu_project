from __future__ import annotations

import argparse
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

BESS_PARAM_PATH = DATA_DIR / "parameters" / "bess.json"
MARKET_PARAM_PATH = DATA_DIR / "parameters" / "market.json"

LOAD_FORECAST_DIR = DATA_DIR / "output_data" / "forecast" / "load"
PV_FORECAST_DIR = DATA_DIR / "output_data" / "forecast" / "pv"
DAM_FORECAST_DIR = DATA_DIR / "output_data" / "forecast" / "dam"

OUTPUT_DIR = DATA_DIR / "output_data" / "schedules" / "da"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Day-ahead BESS optimization"
    )
    parser.add_argument(
        "--day",
        required=True,
        type=str,
        help="Delivery day, e.g. 2026-07-07",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_forecast(path, value_column):
    df = pd.read_excel(path)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
    )

    return df[
        [
            "timestamp",
            "timestamp_utc",
            value_column,
        ]
    ].copy()


def load_forecasts(day):
    load_path = LOAD_FORECAST_DIR / f"load_forecast_{day}.xlsx"
    pv_path = PV_FORECAST_DIR / f"pv_forecast_{day}.xlsx"
    dam_path = DAM_FORECAST_DIR / f"dam_forecast_{day}.xlsx"

    for path in [load_path, pv_path, dam_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Forecast file not found:\n{path}"
            )

    load_df = read_forecast(
        load_path,
        "load_forecast_kw",
    )

    pv_df = read_forecast(
        pv_path,
        "pv_forecast_kw",
    )

    dam_df = read_forecast(
        dam_path,
        "dam_price_forecast",
    )

    df = (
        load_df[
            [
                "timestamp",
                "timestamp_utc",
                "load_forecast_kw",
            ]
        ]
        .merge(
            pv_df[
                [
                    "timestamp_utc",
                    "pv_forecast_kw",
                ]
            ],
            on="timestamp_utc",
            how="inner",
        )
        .merge(
            dam_df[
                [
                    "timestamp_utc",
                    "dam_price_forecast",
                ]
            ],
            on="timestamp_utc",
            how="inner",
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "Forecast files could not be joined."
        )

    return df


def optimize_day(
    forecasts,
    bess,
    market,
):
    load_kw = forecasts["load_forecast_kw"].to_numpy(dtype=float)
    pv_kw = forecasts["pv_forecast_kw"].to_numpy(dtype=float)
    price_eur_mwh = forecasts["dam_price_forecast"].to_numpy(dtype=float)

    T = len(forecasts)

    dt = float(market["dt_hours"])
    grid_import_limit = float(market["grid_import_limit_kw"])
    grid_export_limit = float(market["grid_export_limit_kw"])

    battery_power = float(bess["power_kw"])
    battery_energy = float(bess["energy_kwh"])

    soc_min = float(bess["soc_min"])
    soc_max = float(bess["soc_max"])
    soc_initial = float(bess["soc_initial"])
    soc_final = float(bess["soc_final"])

    eta_charge = float(bess["charge_efficiency"])
    eta_discharge = float(bess["discharge_efficiency"])

    degradation_cost = float(
        bess["degradation_cost_eur_per_kwh"]
    )

    charge_kw = cp.Variable(T, nonneg=True)
    discharge_kw = cp.Variable(T, nonneg=True)

    battery_energy_kwh = cp.Variable(T + 1)

    grid_import_kw = cp.Variable(T, nonneg=True)
    grid_export_kw = cp.Variable(T, nonneg=True)

    pv_curtailment_kw = cp.Variable(T, nonneg=True)

    battery_charge_mode = cp.Variable(T, boolean=True)
    grid_import_mode = cp.Variable(T, boolean=True)

    constraints = []

    min_energy = soc_min * battery_energy
    max_energy = soc_max * battery_energy
    initial_energy = soc_initial * battery_energy
    final_energy = soc_final * battery_energy

    constraints += [
        battery_energy_kwh[0] == initial_energy,
        battery_energy_kwh[T] == final_energy,
        battery_energy_kwh >= min_energy,
        battery_energy_kwh <= max_energy,
    ]

    for t in range(T):

        constraints.append(
            battery_energy_kwh[t + 1]
            ==
            battery_energy_kwh[t]
            + eta_charge * charge_kw[t] * dt
            - discharge_kw[t] * dt / eta_discharge
        )

        constraints.append(
            charge_kw[t]
            <= battery_power * battery_charge_mode[t]
        )

        constraints.append(
            discharge_kw[t]
            <= battery_power * (1 - battery_charge_mode[t])
        )

        constraints.append(
            grid_import_kw[t]
            <= grid_import_limit * grid_import_mode[t]
        )

        constraints.append(
            grid_export_kw[t]
            <= grid_export_limit * (1 - grid_import_mode[t])
        )

        constraints.append(
            pv_curtailment_kw[t] <= pv_kw[t]
        )

        constraints.append(
            grid_import_kw[t]
            + pv_kw[t]
            + discharge_kw[t]
            ==
            load_kw[t]
            + charge_kw[t]
            + grid_export_kw[t]
            + pv_curtailment_kw[t]
        )

    grid_cost_eur = cp.sum(
        cp.multiply(
            price_eur_mwh,
            (
                grid_import_kw
                - grid_export_kw
            )
            * dt
            / 1000.0
        )
    )

    degradation_cost_eur = (
        degradation_cost
        * cp.sum(
            discharge_kw * dt
        )
    )

    objective = cp.Minimize(
        grid_cost_eur
        + degradation_cost_eur
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    if "CBC" not in cp.installed_solvers():
        raise RuntimeError(
            "CBC solver is not available.\n"
            "Install it with:\n"
            "pip install cylp"
        )

    print("\nSolving DA BESS optimization...")

    problem.solve(
        solver=cp.CBC,
        verbose=False,
    )

    if problem.status not in {
        "optimal",
        "optimal_inaccurate",
    }:
        raise RuntimeError(
            f"Optimization failed. Status: {problem.status}"
        )

    result = forecasts.copy()

    result["bess_charge_kw"] = charge_kw.value
    result["bess_discharge_kw"] = discharge_kw.value

    result["bess_net_kw"] = (
        result["bess_discharge_kw"]
        - result["bess_charge_kw"]
    )

    result["bess_energy_kwh"] = (
        battery_energy_kwh.value[:-1]
    )

    result["bess_soc_percent"] = (
        result["bess_energy_kwh"]
        / battery_energy
        * 100.0
    )

    result["grid_import_kw"] = grid_import_kw.value
    result["grid_export_kw"] = grid_export_kw.value

    result["grid_net_kw"] = (
        result["grid_import_kw"]
        - result["grid_export_kw"]
    )

    result["pv_curtailment_kw"] = pv_curtailment_kw.value

    result["grid_energy_cost_eur"] = (
        result["dam_price_forecast"]
        * result["grid_net_kw"]
        * dt
        / 1000.0
    )

    result["degradation_cost_eur"] = (
        degradation_cost
        * result["bess_discharge_kw"]
        * dt
    )

    result["total_step_cost_eur"] = (
        result["grid_energy_cost_eur"]
        + result["degradation_cost_eur"]
    )

    end_energy = float(
        battery_energy_kwh.value[-1]
    )

    end_soc = (
        end_energy
        / battery_energy
    )

    summary = {
        "objective_eur": float(problem.value),
        "load_mwh": float(load_kw.sum() * dt / 1000.0),
        "pv_mwh": float(pv_kw.sum() * dt / 1000.0),

        "grid_import_mwh": float(
            grid_import_kw.value.sum()
            * dt
            / 1000.0
        ),

        "grid_export_mwh": float(
            grid_export_kw.value.sum()
            * dt
            / 1000.0
        ),

        "battery_charge_mwh": float(
            charge_kw.value.sum()
            * dt
            / 1000.0
        ),

        "battery_discharge_mwh": float(
            discharge_kw.value.sum()
            * dt
            / 1000.0
        ),

        "pv_curtailment_mwh": float(
            pv_curtailment_kw.value.sum()
            * dt
            / 1000.0
        ),

        "start_soc_percent": soc_initial * 100.0,
        "end_soc_percent": end_soc * 100.0,
    }

    return result, summary


def save_schedule(
    result,
    summary,
    day,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"da_schedule_{day}.xlsx"
    )

    excel_result = result.copy()

    for column in [
        "timestamp",
        "timestamp_utc",
    ]:
        if column in excel_result.columns:
            excel_result[column] = (
                excel_result[column]
                .astype(str)
            )

    summary_df = pd.DataFrame(
        {
            "metric": list(summary.keys()),
            "value": list(summary.values()),
        }
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        excel_result.to_excel(
            writer,
            sheet_name="schedule",
            index=False,
        )

        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

    print(
        f"\nDA schedule saved:\n  {output_path}"
    )


def main():
    args = parse_args()
    day = args.day

    print(
        "\n"
        "============================================\n"
        "DAY-AHEAD BESS OPTIMIZATION\n"
        "============================================"
    )

    print(f"\nDelivery day: {day}")

    bess = load_json(
        BESS_PARAM_PATH
    )

    market = load_json(
        MARKET_PARAM_PATH
    )

    print("\nBESS:")
    print(f"  Power:  {bess['power_kw']:.0f} kW")
    print(f"  Energy: {bess['energy_kwh']:.0f} kWh")
    print(
        f"  SOC:    "
        f"{bess['soc_min'] * 100:.0f}%"
        f" - "
        f"{bess['soc_max'] * 100:.0f}%"
    )

    print("\nGrid:")
    print(
        f"  Import limit: "
        f"{market['grid_import_limit_kw']:.0f} kW"
    )
    print(
        f"  Export limit: "
        f"{market['grid_export_limit_kw']:.0f} kW"
    )

    forecasts = load_forecasts(
        day
    )

    print(
        f"\nForecast rows: {len(forecasts)}"
    )

    result, summary = optimize_day(
        forecasts=forecasts,
        bess=bess,
        market=market,
    )

    print(
        "\n"
        "============================================\n"
        "OPTIMIZATION SUMMARY\n"
        "============================================"
    )

    print(
        f"Objective cost:       "
        f"{summary['objective_eur']:.2f} EUR"
    )
    print(
        f"Load:                 "
        f"{summary['load_mwh']:.2f} MWh"
    )
    print(
        f"PV:                   "
        f"{summary['pv_mwh']:.2f} MWh"
    )
    print(
        f"Grid import:          "
        f"{summary['grid_import_mwh']:.2f} MWh"
    )
    print(
        f"Grid export:          "
        f"{summary['grid_export_mwh']:.2f} MWh"
    )
    print(
        f"Battery charge:       "
        f"{summary['battery_charge_mwh']:.2f} MWh"
    )
    print(
        f"Battery discharge:    "
        f"{summary['battery_discharge_mwh']:.2f} MWh"
    )
    print(
        f"PV curtailment:       "
        f"{summary['pv_curtailment_mwh']:.2f} MWh"
    )
    print(
        f"Start SOC:            "
        f"{summary['start_soc_percent']:.1f}%"
    )
    print(
        f"End SOC:              "
        f"{summary['end_soc_percent']:.1f}%"
    )

    save_schedule(
        result=result,
        summary=summary,
        day=day,
    )


if __name__ == "__main__":
    main()
