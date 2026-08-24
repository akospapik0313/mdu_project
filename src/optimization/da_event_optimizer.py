from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd


TIMEZONE = "Europe/Stockholm"


def _to_stockholm(value):
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize(TIMEZONE)

    return ts.tz_convert(TIMEZONE)


def _build_event_profiles(
    timestamps,
    events,
):
    """
    Build deterministic logistics profile and collect flexible EV events.
    """

    logistics_kw = np.zeros(
        len(timestamps),
        dtype=float,
    )

    ev_events = []

    for event in events:

        if event["event_type"] == "logistics_load":

            start = _to_stockholm(
                event["start_time"]
            )

            end = _to_stockholm(
                event["end_time"]
            )

            mask = (
                (timestamps >= start)
                &
                (timestamps < end)
            )

            logistics_kw[
                np.asarray(mask)
            ] += float(
                event["power_kw"]
            )

        elif event["event_type"] == "company_ev_charging":

            ev_events.append(
                event
            )

    return logistics_kw, ev_events


def optimize_da_with_events(
    da_schedule,
    normalized_events,
    bess,
    market,
    event_policy,
):
    """
    PRE-GATE day-ahead rescheduling.

    The original DA schedule is NOT treated as a fixed commitment.

    We reuse its forecast inputs:
        - load_forecast_kw
        - pv_forecast_kw
        - dam_price_forecast

    Then we solve the complete delivery day again with the new
    operational events included.

    No intraday prices are used in this function.
    """

    df = da_schedule.copy()

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
    )

    df["timestamp"] = (
        df["timestamp_utc"]
        .dt.tz_convert(TIMEZONE)
    )

    df = (
        df
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    timestamps = df["timestamp"]

    n = len(df)

    dt = float(
        market["dt_hours"]
    )

    load_kw = df[
        "load_forecast_kw"
    ].to_numpy(dtype=float)

    pv_kw = df[
        "pv_forecast_kw"
    ].to_numpy(dtype=float)

    dam_price = df[
        "dam_price_forecast"
    ].to_numpy(dtype=float)

    # ========================================================
    # EVENTS
    # ========================================================

    logistics_kw, ev_events = _build_event_profiles(
        timestamps=timestamps,
        events=normalized_events["events"],
    )

    # ========================================================
    # PHYSICAL PARAMETERS
    # ========================================================

    battery_power_kw = float(
        bess["power_kw"]
    )

    battery_energy_kwh = float(
        bess["energy_kwh"]
    )

    soc_min = float(
        bess["soc_min"]
    )

    soc_max = float(
        bess["soc_max"]
    )

    soc_initial = float(
        bess["soc_initial"]
    )

    soc_final = float(
        bess["soc_final"]
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

    grid_import_limit_kw = float(
        market[
            "grid_import_limit_kw"
        ]
    )

    grid_export_limit_kw = float(
        market[
            "grid_export_limit_kw"
        ]
    )

    min_energy_kwh = (
        soc_min
        * battery_energy_kwh
    )

    max_energy_kwh = (
        soc_max
        * battery_energy_kwh
    )

    initial_energy_kwh = (
        soc_initial
        * battery_energy_kwh
    )

    final_energy_kwh = (
        soc_final
        * battery_energy_kwh
    )

    # ========================================================
    # DECISION VARIABLES
    # ========================================================

    charge_kw = cp.Variable(
        n,
        nonneg=True,
    )

    discharge_kw = cp.Variable(
        n,
        nonneg=True,
    )

    energy_kwh = cp.Variable(
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

    critical_unserved_kw = cp.Variable(
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

    constraints = []

    # ========================================================
    # COMPANY EV FLEXIBLE ENERGY
    # ========================================================

    ev_power_variables = []
    ev_unserved_variables = []

    total_ev_requested_kwh = 0.0

    for event_index, event in enumerate(
        ev_events
    ):

        ev_power_kw = cp.Variable(
            n,
            nonneg=True,
            name=f"ev_power_{event_index}",
        )

        ev_unserved_kwh = cp.Variable(
            nonneg=True,
            name=f"ev_unserved_{event_index}",
        )

        start = _to_stockholm(
            event["arrival_time"]
        )

        end = _to_stockholm(
            event["departure_time"]
        )

        active_mask = np.asarray(
            (
                (timestamps >= start)
                &
                (timestamps < end)
            )
        )

        max_power_kw = float(
            event[
                "total_max_power_kw"
            ]
        )

        required_energy_kwh = float(
            event[
                "total_energy_required_kwh"
            ]
        )

        total_ev_requested_kwh += (
            required_energy_kwh
        )

        for t in range(n):

            if active_mask[t]:

                constraints.append(
                    ev_power_kw[t]
                    <= max_power_kw
                )

            else:

                constraints.append(
                    ev_power_kw[t]
                    == 0
                )

        constraints.append(
            cp.sum(
                ev_power_kw
            )
            * dt
            +
            ev_unserved_kwh
            ==
            required_energy_kwh
        )

        ev_power_variables.append(
            ev_power_kw
        )

        ev_unserved_variables.append(
            ev_unserved_kwh
        )

    if ev_power_variables:

        total_ev_kw = sum(
            ev_power_variables
        )

        total_ev_unserved_kwh = sum(
            ev_unserved_variables
        )

    else:

        total_ev_kw = np.zeros(
            n,
            dtype=float,
        )

        total_ev_unserved_kwh = 0.0

    # ========================================================
    # BESS + GRID CONSTRAINTS
    # ========================================================

    constraints += [
        energy_kwh[0]
        == initial_energy_kwh,

        energy_kwh[n]
        == final_energy_kwh,

        energy_kwh
        >= min_energy_kwh,

        energy_kwh
        <= max_energy_kwh,
    ]

    for t in range(n):

        constraints.append(
            energy_kwh[t + 1]
            ==
            energy_kwh[t]
            +
            eta_charge
            * charge_kw[t]
            * dt
            -
            discharge_kw[t]
            * dt
            / eta_discharge
        )

        # BESS charge/discharge mutually exclusive.
        constraints.append(
            charge_kw[t]
            <=
            battery_power_kw
            * battery_charge_mode[t]
        )

        constraints.append(
            discharge_kw[t]
            <=
            battery_power_kw
            * (
                1
                - battery_charge_mode[t]
            )
        )

        # Grid import/export mutually exclusive.
        constraints.append(
            grid_import_kw[t]
            <=
            grid_import_limit_kw
            * grid_import_mode[t]
        )

        constraints.append(
            grid_export_kw[t]
            <=
            grid_export_limit_kw
            * (
                1
                - grid_import_mode[t]
            )
        )

        constraints.append(
            pv_curtailment_kw[t]
            <= pv_kw[t]
        )

        # Emergency slack may only cover critical logistics.
        constraints.append(
            critical_unserved_kw[t]
            <= logistics_kw[t]
        )

        # Site power balance.
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
            logistics_kw[t]
            +
            total_ev_kw[t]
            +
            charge_kw[t]
            +
            grid_export_kw[t]
            +
            pv_curtailment_kw[t]
        )

    # ========================================================
    # OBJECTIVE
    # ========================================================

    grid_net_kw = (
        grid_import_kw
        -
        grid_export_kw
    )

    dam_energy_cost_eur = cp.sum(
        cp.multiply(
            dam_price,
            grid_net_kw
            * dt
            / 1000.0,
        )
    )

    degradation_cost_eur = (
        degradation_cost
        *
        cp.sum(
            discharge_kw
            * dt
        )
    )

    ev_unserved_penalty = float(
        event_policy[
            "company_ev"
        ][
            "unserved_penalty_eur_per_kwh"
        ]
    )

    critical_unserved_penalty = float(
        event_policy[
            "logistics"
        ][
            "critical_unserved_penalty_eur_per_kwh"
        ]
    )

    critical_unserved_energy_kwh = (
        cp.sum(
            critical_unserved_kw
        )
        * dt
    )

    objective = cp.Minimize(
        dam_energy_cost_eur
        +
        degradation_cost_eur
        +
        ev_unserved_penalty
        * total_ev_unserved_kwh
        +
        critical_unserved_penalty
        * critical_unserved_energy_kwh
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    # ========================================================
    # ROBUST MILP SOLVE + FEASIBILITY CHECK
    # ========================================================
    #
    # Solver status alone is not enough for an operational
    # schedule. The returned primal values are checked against
    # the model constraints before the schedule is accepted.
    # ========================================================

    installed_solvers = set(
        cp.installed_solvers()
    )

    preferred_solvers = [
        "HIGHS",
        "SCIPY",
        "GLPK_MI",
        "SCIP",
        "CBC",
    ]

    candidate_solvers = [
        solver
        for solver in preferred_solvers
        if solver in installed_solvers
    ]

    if not candidate_solvers:

        raise RuntimeError(
            "No mixed-integer solver is available. "
            "Install HiGHS with: pip install highspy"
        )

    feasibility_tolerance = 1e-3

    def maximum_constraint_violation():

        max_violation = 0.0

        for constraint in problem.constraints:

            try:
                violation = constraint.violation()

            except Exception:
                return float("inf")

            if violation is None:
                return float("inf")

            violation_array = np.asarray(
                violation,
                dtype=float,
            )

            if violation_array.size == 0:
                continue

            if not np.all(
                np.isfinite(
                    violation_array
                )
            ):
                return float("inf")

            max_violation = max(
                max_violation,
                float(
                    np.max(
                        np.abs(
                            violation_array
                        )
                    )
                ),
            )

        return max_violation

    def maximum_physical_violation():

        required_values = [
            charge_kw.value,
            discharge_kw.value,
            energy_kwh.value,
            grid_import_kw.value,
            grid_export_kw.value,
            pv_curtailment_kw.value,
            critical_unserved_kw.value,
        ]

        if any(
            value is None
            for value in required_values
        ):
            return float("inf")

        charge_check = np.asarray(
            charge_kw.value,
            dtype=float,
        )

        discharge_check = np.asarray(
            discharge_kw.value,
            dtype=float,
        )

        energy_check = np.asarray(
            energy_kwh.value,
            dtype=float,
        )

        import_check = np.asarray(
            grid_import_kw.value,
            dtype=float,
        )

        export_check = np.asarray(
            grid_export_kw.value,
            dtype=float,
        )

        checks = []

        # SOC / energy bounds.
        checks.append(
            max(
                0.0,
                min_energy_kwh
                -
                float(
                    np.min(
                        energy_check
                    )
                ),
            )
        )

        checks.append(
            max(
                0.0,
                float(
                    np.max(
                        energy_check
                    )
                )
                -
                max_energy_kwh,
            )
        )

        # Initial / final energy.
        checks.append(
            abs(
                float(
                    energy_check[0]
                )
                -
                initial_energy_kwh
            )
        )

        checks.append(
            abs(
                float(
                    energy_check[-1]
                )
                -
                final_energy_kwh
            )
        )

        # BESS state transition.
        expected_next_energy = (
            energy_check[:-1]
            +
            eta_charge
            * charge_check
            * dt
            -
            discharge_check
            * dt
            / eta_discharge
        )

        checks.append(
            float(
                np.max(
                    np.abs(
                        energy_check[1:]
                        -
                        expected_next_energy
                    )
                )
            )
        )

        # Power limits.
        checks.append(
            max(
                0.0,
                float(
                    np.max(
                        charge_check
                    )
                )
                -
                battery_power_kw,
            )
        )

        checks.append(
            max(
                0.0,
                float(
                    np.max(
                        discharge_check
                    )
                )
                -
                battery_power_kw,
            )
        )

        checks.append(
            max(
                0.0,
                float(
                    np.max(
                        import_check
                    )
                )
                -
                grid_import_limit_kw,
            )
        )

        checks.append(
            max(
                0.0,
                float(
                    np.max(
                        export_check
                    )
                )
                -
                grid_export_limit_kw,
            )
        )

        # No simultaneous charge/discharge.
        checks.append(
            float(
                np.max(
                    np.minimum(
                        charge_check,
                        discharge_check,
                    )
                )
            )
        )

        # No simultaneous import/export.
        checks.append(
            float(
                np.max(
                    np.minimum(
                        import_check,
                        export_check,
                    )
                )
            )
        )

        return max(
            checks
        )

    solved = False
    solver_errors = []

    print(
        "\nSolving PRE-GATE day-ahead rescheduling..."
    )

    print(
        "Available solver candidates: "
        +
        ", ".join(
            candidate_solvers
        )
    )

    for solver_name in candidate_solvers:

        print(
            f"Trying solver: {solver_name}"
        )

        try:
            problem.solve(
                solver=solver_name,
                verbose=False,
            )

        except Exception as exc:

            solver_errors.append(
                f"{solver_name}: {exc}"
            )

            print(
                f"  -> solver failed: {exc}"
            )

            continue

        if problem.status not in {
            "optimal",
            "optimal_inaccurate",
        }:

            solver_errors.append(
                f"{solver_name}: status={problem.status}"
            )

            print(
                f"  -> rejected status: "
                f"{problem.status}"
            )

            continue

        constraint_violation = (
            maximum_constraint_violation()
        )

        physical_violation = (
            maximum_physical_violation()
        )

        print(
            f"  -> status: {problem.status}"
        )

        print(
            "  -> max model constraint violation: "
            f"{constraint_violation:.6g}"
        )

        print(
            "  -> max physical violation: "
            f"{physical_violation:.6g}"
        )

        if (
            constraint_violation
            <= feasibility_tolerance
            and
            physical_violation
            <= feasibility_tolerance
        ):

            solved = True

            print(
                "Accepted feasible solution from "
                f"{solver_name}."
            )

            break

        solver_errors.append(
            (
                f"{solver_name}: infeasible primal values "
                f"(constraint violation="
                f"{constraint_violation:.6g}, "
                f"physical violation="
                f"{physical_violation:.6g})"
            )
        )

        print(
            "  -> rejected: returned values are "
            "not physically feasible."
        )

    if not solved:

        details = "\n".join(
            f"  - {message}"
            for message in solver_errors
        )

        raise RuntimeError(
            "No solver returned a physically valid "
            "day-ahead schedule.\n"
            f"{details}\n\n"
            "Recommended fix: install HiGHS with "
            "`pip install highspy` and rerun."
        )


    # ========================================================
    # RESULTS
    # ========================================================

    charge_value = np.asarray(
        charge_kw.value,
        dtype=float,
    )

    discharge_value = np.asarray(
        discharge_kw.value,
        dtype=float,
    )

    energy_value = np.asarray(
        energy_kwh.value,
        dtype=float,
    )

    import_value = np.asarray(
        grid_import_kw.value,
        dtype=float,
    )

    export_value = np.asarray(
        grid_export_kw.value,
        dtype=float,
    )

    curtailment_value = np.asarray(
        pv_curtailment_kw.value,
        dtype=float,
    )

    critical_unserved_value = np.asarray(
        critical_unserved_kw.value,
        dtype=float,
    )

    if ev_power_variables:

        company_ev_kw = np.sum(
            [
                np.asarray(
                    variable.value,
                    dtype=float,
                )
                for variable
                in ev_power_variables
            ],
            axis=0,
        )

        company_ev_unserved_kwh = float(
            sum(
                float(
                    variable.value
                )
                for variable
                in ev_unserved_variables
            )
        )

    else:

        company_ev_kw = np.zeros(
            n,
            dtype=float,
        )

        company_ev_unserved_kwh = 0.0

    company_ev_delivered_kwh = max(
        0.0,
        total_ev_requested_kwh
        -
        company_ev_unserved_kwh,
    )

    # ========================================================
    # PER-EVENT DA POWER PROFILES
    # ========================================================
    #
    # These profiles let the ID stage preserve the exact
    # event schedule selected in the DA stage.
    # ========================================================

    for event_index, event in enumerate(
        ev_events
    ):

        event_id = event.get(
            "event_id",
            f"ev_{event_index}",
        )

        df[
            f"event_profile_{event_id}_kw"
        ] = np.asarray(
            ev_power_variables[
                event_index
            ].value,
            dtype=float,
        )

    for event_index, event in enumerate(
        normalized_events[
            "events"
        ]
    ):

        if (
            event.get(
                "event_type"
            )
            !=
            "logistics_load"
        ):
            continue

        event_id = event.get(
            "event_id",
            f"logistics_{event_index}",
        )

        start = _to_stockholm(
            event[
                "start_time"
            ]
        )

        end = _to_stockholm(
            event[
                "end_time"
            ]
        )

        profile = np.zeros(
            n,
            dtype=float,
        )

        profile_mask = np.asarray(
            (
                (timestamps >= start)
                &
                (timestamps < end)
            )
        )

        profile[
            profile_mask
        ] = float(
            event[
                "power_kw"
            ]
        )

        df[
            f"event_profile_{event_id}_kw"
        ] = profile

    df["logistics_event_kw"] = (
        logistics_kw
    )

    df["company_ev_kw"] = (
        company_ev_kw
    )

    df["critical_unserved_kw"] = (
        critical_unserved_value
    )

    df["effective_load_kw"] = (
        load_kw
        +
        logistics_kw
        +
        company_ev_kw
    )

    df["bess_charge_kw"] = (
        charge_value
    )

    df["bess_discharge_kw"] = (
        discharge_value
    )

    df["bess_net_kw"] = (
        discharge_value
        -
        charge_value
    )

    df["bess_energy_kwh"] = (
        energy_value[:-1]
    )

    df["bess_soc_percent"] = (
        energy_value[:-1]
        /
        battery_energy_kwh
        * 100.0
    )

    df["grid_import_kw"] = (
        import_value
    )

    df["grid_export_kw"] = (
        export_value
    )

    df["grid_net_kw"] = (
        import_value
        -
        export_value
    )

    df["pv_curtailment_kw"] = (
        curtailment_value
    )

    df["grid_energy_cost_eur"] = (
        dam_price
        *
        df[
            "grid_net_kw"
        ].to_numpy(dtype=float)
        *
        dt
        /
        1000.0
    )

    df["degradation_cost_eur"] = (
        degradation_cost
        *
        discharge_value
        *
        dt
    )

    df["total_step_cost_eur"] = (
        df["grid_energy_cost_eur"]
        +
        df["degradation_cost_eur"]
    )

    critical_unserved_kwh = float(
        np.sum(
            critical_unserved_value
        )
        * dt
    )

    summary = {
        "market_stage":
            "DA",

        "objective_eur":
            float(problem.value),

        "dam_energy_cost_eur":
            float(
                np.sum(
                    df[
                        "grid_energy_cost_eur"
                    ]
                )
            ),

        "degradation_cost_eur":
            float(
                np.sum(
                    df[
                        "degradation_cost_eur"
                    ]
                )
            ),

        "company_ev_requested_kwh":
            float(
                total_ev_requested_kwh
            ),

        "company_ev_delivered_kwh":
            float(
                company_ev_delivered_kwh
            ),

        "company_ev_unserved_kwh":
            float(
                company_ev_unserved_kwh
            ),

        "critical_logistics_unserved_kwh":
            float(
                critical_unserved_kwh
            ),

        "final_soc_percent":
            float(
                energy_value[-1]
                /
                battery_energy_kwh
                * 100.0
            ),
    }

    return df, summary