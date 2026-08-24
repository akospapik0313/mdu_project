from __future__ import annotations

import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


TIMEZONE = "Europe/Stockholm"


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def parse_as_of(value):
    ts = pd.Timestamp(
        value
    )

    if ts.tzinfo is None:
        ts = ts.tz_localize(
            TIMEZONE
        )
    else:
        ts = ts.tz_convert(
            TIMEZONE
        )

    return ts


def build_event_profiles(
    timestamps,
    events,
):
    logistics_kw = np.zeros(
        len(timestamps),
        dtype=float,
    )

    ev_events = []

    for event in events:
        if (
            event["event_type"]
            == "logistics_load"
        ):
            start = pd.Timestamp(
                event["start_time"]
            ).tz_convert(
                TIMEZONE
            )

            end = pd.Timestamp(
                event["end_time"]
            ).tz_convert(
                TIMEZONE
            )

            mask = (
                (timestamps >= start)
                &
                (timestamps < end)
            )

            logistics_kw[
                mask
            ] += float(
                event["power_kw"]
            )

        elif (
            event["event_type"]
            == "company_ev_charging"
        ):
            ev_events.append(
                event
            )

    return (
        logistics_kw,
        ev_events,
    )


def optimize_intraday(
    da_schedule,
    id_prices_eur_mwh,
    normalized_events,
    as_of,
    bess,
    market,
    event_policy,
    history_schedule=None,
):
    """
    Re-optimization from --as-of onward.

    Past:
        fixed exactly as the DA schedule.

    Future:
        can change BESS/grid schedule.

    The original DA grid schedule remains the MARKET commitment.

    Physical demand is rebuilt from:
        base load forecast
        + CURRENT active logistics events
        + CURRENT active flexible EV events

    This separation is intentional:

        commitment = DA grid_net_kw
        physical state = current event ledger

    Therefore a post-gate CANCEL or UPDATE of a pre-gate event
    can be represented correctly.

    ID trade = new grid net - DA committed grid net.

    history_schedule:
        Optional latest ID-rescheduled schedule for this delivery day.

        Rows before the new --as-of are preserved from this schedule
        so already executed operational actions remain visible and
        physically consistent in the current schedule.

        Rows from --as-of onward are re-optimized.

        The DA schedule is still used separately as the MARKET
        commitment for ID settlement.
    """

    commitment_df = da_schedule.copy()

    if history_schedule is not None:
        df = history_schedule.copy()
    else:
        df = da_schedule.copy()

    # Continuous optimization results must remain float columns.
    float_result_columns = [
        "bess_charge_kw",
        "bess_discharge_kw",
        "bess_net_kw",
        "bess_energy_kwh",
        "bess_soc_percent",
        "grid_import_kw",
        "grid_export_kw",
        "grid_net_kw",
        "pv_curtailment_kw",
    ]

    for column in float_result_columns:

        if column in df.columns:

            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
                .astype(float)
            )

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

    # DA commitment is kept separately from the physical/history
    # schedule used for output.
    commitment_df["timestamp_utc"] = pd.to_datetime(
        commitment_df["timestamp_utc"],
        utc=True,
    )

    commitment_df = (
        commitment_df
        .sort_values(
            "timestamp_utc"
        )
        .reset_index(
            drop=True
        )
    )

    if not np.array_equal(
        df["timestamp_utc"].to_numpy(),
        commitment_df["timestamp_utc"].to_numpy(),
    ):
        raise ValueError(
            "History schedule and DA commitment timestamps do not align."
        )

    timestamps = df[
        "timestamp"
    ]

    as_of = parse_as_of(
        as_of
    )

    # First fully controllable 15-minute slot.
    effective_as_of = as_of.ceil(
        "15min"
    )

    future_mask = (
        timestamps
        >= effective_as_of
    )

    future_indices = np.flatnonzero(
        future_mask.to_numpy()
    )

    if len(
        future_indices
    ) == 0:
        raise ValueError(
            "No future intervals remain after --as-of."
        )

    first_future = int(
        future_indices[0]
    )

    # ========================================================
    # ACTIVE EVENTS VS DA EVENT PROFILES
    # ========================================================

    active_events = normalized_events[
        "events"
    ]

    fixed_company_ev_kw = np.zeros(
        len(df),
        dtype=float,
    )

    fixed_logistics_kw = np.zeros(
        len(df),
        dtype=float,
    )

    new_id_events = []

    active_total_requested_ev_kwh = 0.0

    for event in active_events:

        event_id = event.get(
            "event_id"
        )

        profile_column = (
            f"event_profile_{event_id}_kw"
            if event_id
            else None
        )

        if (
            event.get(
                "event_type"
            )
            ==
            "company_ev_charging"
        ):

            active_total_requested_ev_kwh += float(
                event[
                    "total_energy_required_kwh"
                ]
            )

        if (
            profile_column
            and
            profile_column
            in commitment_df.columns
        ):

            profile = (
                pd.to_numeric(
                    commitment_df[
                        profile_column
                    ],
                    errors="coerce",
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=float
                )
            )

            if (
                event.get(
                    "event_type"
                )
                ==
                "company_ev_charging"
            ):

                fixed_company_ev_kw += profile

            elif (
                event.get(
                    "event_type"
                )
                ==
                "logistics_load"
            ):

                fixed_logistics_kw += profile

        else:

            new_id_events.append(
                event
            )

    new_logistics_kw, ev_events = (
        build_event_profiles(
            timestamps,
            new_id_events,
        )
    )

    # --------------------------------------------------------
    # Full-day reporting columns
    # --------------------------------------------------------

    # --------------------------------------------------------
    # MARKET COMMITMENT DIAGNOSTICS
    # --------------------------------------------------------

    if "logistics_event_kw" in commitment_df.columns:

        df["da_logistics_event_kw"] = (
            pd.to_numeric(
                commitment_df["logistics_event_kw"],
                errors="coerce",
            )
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )

    else:

        df["da_logistics_event_kw"] = 0.0

    if "company_ev_kw" in commitment_df.columns:

        df["da_company_ev_kw"] = (
            pd.to_numeric(
                commitment_df["company_ev_kw"],
                errors="coerce",
            )
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )

    else:

        df["da_company_ev_kw"] = 0.0

    # --------------------------------------------------------
    # PHYSICAL EVENT HISTORY
    # --------------------------------------------------------
    #
    # Past rows stay exactly as they were in the latest ID schedule.
    # Future rows are rebuilt from the CURRENT active ledger.
    # --------------------------------------------------------

    if "logistics_event_kw" not in df.columns:
        df["logistics_event_kw"] = 0.0
    else:
        df["logistics_event_kw"] = (
            pd.to_numeric(
                df["logistics_event_kw"],
                errors="coerce",
            )
            .fillna(0.0)
            .astype(float)
        )

    if "company_ev_kw" not in df.columns:
        df["company_ev_kw"] = 0.0
    else:
        df["company_ev_kw"] = (
            pd.to_numeric(
                df["company_ev_kw"],
                errors="coerce",
            )
            .fillna(0.0)
            .astype(float)
        )

    if "critical_unserved_kw" not in df.columns:
        df["critical_unserved_kw"] = 0.0
    else:
        df["critical_unserved_kw"] = (
            pd.to_numeric(
                df["critical_unserved_kw"],
                errors="coerce",
            )
            .fillna(0.0)
            .astype(float)
        )

    # Future physical state starts from active DA event profiles.
    # New ID-stage events are added by the MILP.
    df.loc[
        future_mask,
        "logistics_event_kw",
    ] = (
        fixed_logistics_kw[
            future_indices
        ]
        +
        new_logistics_kw[
            future_indices
        ]
    )

    df.loc[
        future_mask,
        "company_ev_kw",
    ] = fixed_company_ev_kw[
        future_indices
    ]

    df.loc[
        future_mask,
        "critical_unserved_kw",
    ] = 0.0

    df["id_vwap_eur_mwh"] = (
        id_prices_eur_mwh
    )

    # IMPORTANT:
    # ID settlement always compares against the fixed DA commitment,
    # never against the previous ID-rescheduled physical schedule.
    df["da_grid_net_kw"] = (
        pd.to_numeric(
            commitment_df["grid_net_kw"],
            errors="coerce",
        )
        .astype(float)
        .to_numpy()
    )

    # Preserve past ID settlement history if available.
    if "id_trade_kw" not in df.columns:
        df["id_trade_kw"] = 0.0

    if "id_trade_cost_eur" not in df.columns:
        df["id_trade_cost_eur"] = 0.0

    df.loc[
        future_mask,
        "id_trade_kw",
    ] = 0.0

    df.loc[
        future_mask,
        "id_trade_cost_eur",
    ] = 0.0

    # --------------------------------------------------------
    # Future arrays
    # --------------------------------------------------------

    future_df = df.loc[
        future_mask
    ].copy().reset_index(
        drop=True
    )

    future_times = future_df[
        "timestamp"
    ]

    n = len(
        future_df
    )

    dt = float(
        market["dt_hours"]
    )

    load_kw = future_df[
        "load_forecast_kw"
    ].to_numpy(
        dtype=float
    )

    pv_kw = future_df[
        "pv_forecast_kw"
    ].to_numpy(
        dtype=float
    )

    fixed_company_ev_future_kw = (
        fixed_company_ev_kw[
            future_indices
        ]
    )

    fixed_logistics_future_kw = (
        fixed_logistics_kw[
            future_indices
        ]
    )

    new_logistics_future_kw = (
        new_logistics_kw[
            future_indices
        ]
    )

    price = np.asarray(
        id_prices_eur_mwh,
        dtype=float,
    )[
        future_indices
    ]

    da_grid_net_kw = (
        pd.to_numeric(
            commitment_df.loc[
                future_mask,
                "grid_net_kw",
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # BESS parameters
    # --------------------------------------------------------

    battery_power = float(
        bess["power_kw"]
    )

    battery_energy = float(
        bess["energy_kwh"]
    )

    eta_charge = float(
        bess["charge_efficiency"]
    )

    eta_discharge = float(
        bess["discharge_efficiency"]
    )

    degradation_cost = float(
        bess[
            "degradation_cost_eur_per_kwh"
        ]
    )

    min_energy = (
        float(
            bess["soc_min"]
        )
        * battery_energy
    )

    max_energy = (
        float(
            bess["soc_max"]
        )
        * battery_energy
    )

    final_energy = (
        float(
            bess["soc_final"]
        )
        * battery_energy
    )

    # DA schedule stores the battery energy at the beginning
    # of each interval. This becomes the rescheduling initial state.
    initial_energy = float(
        df.loc[
            first_future,
            "bess_energy_kwh",
        ]
    )

    grid_import_limit = float(
        market[
            "grid_import_limit_kw"
        ]
    )

    grid_export_limit = float(
        market[
            "grid_export_limit_kw"
        ]
    )

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    charge_kw = cp.Variable(
        n,
        nonneg=True,
    )

    discharge_kw = cp.Variable(
        n,
        nonneg=True,
    )

    battery_energy_kwh = cp.Variable(
        n + 1
    )

    grid_import_kw = cp.Variable(
        n,
        nonneg=True,
    )

    grid_export_kw = cp.Variable(
        n,
        nonneg=True,
    )

    pv_curtailment_kw = cp.Variable(
        n,
        nonneg=True,
    )

    battery_charge_mode = cp.Variable(
        n,
        boolean=True,
    )

    grid_import_mode = cp.Variable(
        n,
        boolean=True,
    )

    # Emergency slack only for critical logistics demand.
    # It never relaxes grid/BESS limits.
    critical_unserved_kw = cp.Variable(
        n,
        nonneg=True,
    )

    # --------------------------------------------------------
    # Flexible company EV variables
    # --------------------------------------------------------

    ev_power_variables = []
    ev_unserved_variables = []

    constraints = []

    for event_index, event in enumerate(
        ev_events
    ):
        power = cp.Variable(
            n,
            nonneg=True,
            name=f"ev_power_{event_index}",
        )

        unserved = cp.Variable(
            nonneg=True,
            name=f"ev_unserved_{event_index}",
        )

        arrival = pd.Timestamp(
            event["arrival_time"]
        ).tz_convert(
            TIMEZONE
        )

        departure = pd.Timestamp(
            event["departure_time"]
        ).tz_convert(
            TIMEZONE
        )

        active_mask = (
            (future_times >= arrival)
            &
            (future_times < departure)
        ).to_numpy()

        max_power = float(
            event[
                "total_max_power_kw"
            ]
        )

        required_energy = float(
            event[
                "total_energy_required_kwh"
            ]
        )

        for t in range(
            n
        ):
            if active_mask[t]:
                constraints.append(
                    power[t]
                    <= max_power
                )
            else:
                constraints.append(
                    power[t]
                    == 0
                )

        delivered_energy = (
            cp.sum(
                power
            )
            * dt
        )

        constraints.append(
            delivered_energy
            + unserved
            ==
            required_energy
        )

        ev_power_variables.append(
            power
        )

        ev_unserved_variables.append(
            unserved
        )

    if ev_power_variables:
        total_ev_kw = sum(
            ev_power_variables
        )
    else:
        total_ev_kw = np.zeros(
            n,
            dtype=float,
        )

    # --------------------------------------------------------
    # Battery state constraints
    # --------------------------------------------------------

    constraints += [
        battery_energy_kwh[0]
        == initial_energy,

        battery_energy_kwh[n]
        == final_energy,

        battery_energy_kwh
        >= min_energy,

        battery_energy_kwh
        <= max_energy,
    ]

    for t in range(
        n
    ):
        constraints.append(
            battery_energy_kwh[t + 1]
            ==
            battery_energy_kwh[t]
            +
            eta_charge
            * charge_kw[t]
            * dt
            -
            discharge_kw[t]
            * dt
            / eta_discharge
        )

        constraints.append(
            charge_kw[t]
            <=
            battery_power
            * battery_charge_mode[t]
        )

        constraints.append(
            discharge_kw[t]
            <=
            battery_power
            * (
                1
                - battery_charge_mode[t]
            )
        )

        constraints.append(
            grid_import_kw[t]
            <=
            grid_import_limit
            * grid_import_mode[t]
        )

        constraints.append(
            grid_export_kw[t]
            <=
            grid_export_limit
            * (
                1
                - grid_import_mode[t]
            )
        )

        constraints.append(
            pv_curtailment_kw[t]
            <= pv_kw[t]
        )

        # Only logistics demand can be represented as
        # emergency critical-unserved load.
        constraints.append(
            critical_unserved_kw[t]
            <= new_logistics_future_kw[t]
        )

        # Power balance.
        constraints.append(
            grid_import_kw[t]
            +
            pv_kw[t]
            +
            discharge_kw[t]
            +
            critical_unserved_kw[t]
            ==
            load_kw[t]
            +
            fixed_logistics_future_kw[t]
            +
            fixed_company_ev_future_kw[t]
            +
            new_logistics_future_kw[t]
            +
            total_ev_kw[t]
            +
            charge_kw[t]
            +
            grid_export_kw[t]
            +
            pv_curtailment_kw[t]
        )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    new_grid_net_kw = (
        grid_import_kw
        -
        grid_export_kw
    )

    id_trade_kw = (
        new_grid_net_kw
        -
        da_grid_net_kw
    )

    id_trade_cost = cp.sum(
        cp.multiply(
            price,
            id_trade_kw
            * dt
            / 1000.0,
        )
    )

    degradation = (
        degradation_cost
        *
        cp.sum(
            discharge_kw
            * dt
        )
    )

    ev_penalty = float(
        event_policy[
            "company_ev"
        ][
            "unserved_penalty_eur_per_kwh"
        ]
    )

    if ev_unserved_variables:
        ev_unserved_total = sum(
            ev_unserved_variables
        )
    else:
        ev_unserved_total = 0.0

    critical_penalty = float(
        event_policy[
            "logistics"
        ][
            "critical_unserved_penalty_eur_per_kwh"
        ]
    )

    critical_unserved_energy = (
        cp.sum(
            critical_unserved_kw
        )
        * dt
    )

    objective = cp.Minimize(
        id_trade_cost
        +
        degradation
        +
        ev_penalty
        * ev_unserved_total
        +
        critical_penalty
        * critical_unserved_energy
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    if "CBC" not in cp.installed_solvers():
        raise RuntimeError(
            "CBC solver is not available."
        )

    print(
        "\nSolving intraday rescheduling..."
    )

    problem.solve(
        solver=cp.CBC,
        verbose=False,
    )

    if problem.status not in {
        "optimal",
        "optimal_inaccurate",
    }:
        raise RuntimeError(
            f"Intraday optimization failed: "
            f"{problem.status}"
        )

    # --------------------------------------------------------
    # Write future solution into copied DA schedule
    # --------------------------------------------------------

    future_rows = df.index[
        future_mask
    ]

    df.loc[
        future_rows,
        "bess_charge_kw",
    ] = charge_kw.value

    df.loc[
        future_rows,
        "bess_discharge_kw",
    ] = discharge_kw.value

    df.loc[
        future_rows,
        "bess_net_kw",
    ] = (
        discharge_kw.value
        -
        charge_kw.value
    )

    df.loc[
        future_rows,
        "bess_energy_kwh",
    ] = (
        battery_energy_kwh.value[:-1]
    )

    df.loc[
        future_rows,
        "bess_soc_percent",
    ] = (
        battery_energy_kwh.value[:-1]
        /
        battery_energy
        * 100.0
    )

    df.loc[
        future_rows,
        "grid_import_kw",
    ] = grid_import_kw.value

    df.loc[
        future_rows,
        "grid_export_kw",
    ] = grid_export_kw.value

    df.loc[
        future_rows,
        "grid_net_kw",
    ] = (
        grid_import_kw.value
        -
        grid_export_kw.value
    )

    df.loc[
        future_rows,
        "pv_curtailment_kw",
    ] = pv_curtailment_kw.value

    df.loc[
        future_rows,
        "critical_unserved_kw",
    ] = critical_unserved_kw.value

    if ev_power_variables:
        company_ev_future = np.sum(
            [
                variable.value
                for variable
                in ev_power_variables
            ],
            axis=0,
        )
    else:
        company_ev_future = np.zeros(
            n
        )

    df.loc[
        future_rows,
        "company_ev_kw",
    ] = (
        fixed_company_ev_future_kw
        +
        company_ev_future
    )

    if "effective_load_kw" not in df.columns:
        df["effective_load_kw"] = (
            df["load_forecast_kw"]
            +
            df["logistics_event_kw"]
            +
            df["company_ev_kw"]
        )
    else:
        df["effective_load_kw"] = (
            pd.to_numeric(
                df["effective_load_kw"],
                errors="coerce",
            )
            .astype(float)
        )

        df.loc[
            future_mask,
            "effective_load_kw",
        ] = (
            df.loc[
                future_mask,
                "load_forecast_kw",
            ].to_numpy(
                dtype=float
            )
            +
            df.loc[
                future_mask,
                "logistics_event_kw",
            ].to_numpy(
                dtype=float
            )
            +
            df.loc[
                future_mask,
                "company_ev_kw",
            ].to_numpy(
                dtype=float
            )
        )

    df.loc[
        future_rows,
        "id_trade_kw",
    ] = (
        df.loc[
            future_rows,
            "grid_net_kw",
        ].to_numpy()
        -
        da_grid_net_kw
    )

    df.loc[
        future_rows,
        "id_trade_cost_eur",
    ] = (
        price
        *
        df.loc[
            future_rows,
            "id_trade_kw",
        ].to_numpy()
        *
        dt
        /
        1000.0
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    ev_requested_kwh = sum(
        float(
            event[
                "total_energy_required_kwh"
            ]
        )
        for event in ev_events
    )

    ev_unserved_kwh = (
        sum(
            float(
                variable.value
            )
            for variable
            in ev_unserved_variables
        )
        if ev_unserved_variables
        else 0.0
    )

    new_id_ev_delivered_kwh = (
        ev_requested_kwh
        -
        ev_unserved_kwh
    )

    fixed_da_ev_delivered_kwh = float(
        np.sum(
            fixed_company_ev_kw
        )
        * dt
    )

    total_active_ev_delivered_kwh = (
        fixed_da_ev_delivered_kwh
        +
        new_id_ev_delivered_kwh
    )

    total_active_ev_unserved_kwh = max(
        0.0,
        active_total_requested_ev_kwh
        -
        total_active_ev_delivered_kwh,
    )

    critical_unserved_kwh = float(
        np.sum(
            critical_unserved_kw.value
        )
        * dt
    )

    summary = {
        "as_of":
            effective_as_of.isoformat(),

        "objective_eur":
            float(
                problem.value
            ),

        "id_trade_cost_eur":
            float(
                np.sum(
                    df.loc[
                        future_rows,
                        "id_trade_cost_eur",
                    ]
                )
            ),

        "company_ev_requested_kwh":
            float(
                active_total_requested_ev_kwh
            ),

        "company_ev_delivered_kwh":
            float(
                total_active_ev_delivered_kwh
            ),

        "company_ev_unserved_kwh":
            float(
                total_active_ev_unserved_kwh
            ),

        "da_fixed_company_ev_kwh":
            float(
                fixed_da_ev_delivered_kwh
            ),

        "new_id_company_ev_requested_kwh":
            float(
                ev_requested_kwh
            ),

        "new_id_company_ev_delivered_kwh":
            float(
                new_id_ev_delivered_kwh
            ),

        "new_id_company_ev_unserved_kwh":
            float(
                ev_unserved_kwh
            ),

        "critical_logistics_unserved_kwh":
            critical_unserved_kwh,

        "final_soc_percent":
            float(
                battery_energy_kwh.value[-1]
                /
                battery_energy
                * 100.0
            ),
    }

    return (
        df,
        summary,
    )