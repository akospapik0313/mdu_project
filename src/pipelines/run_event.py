from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.agent.event_handler import (
    load_json,
    load_logistics_library,
    logistics_catalog_for_llm,
    normalize_events,
)

from src.agent.event_store import (
    active_events,
    apply_operations,
    events_for_optimizer,
    load_event_ledger,
    reset_event_ledger,
    save_event_ledger,
)

from src.agent.llm_event_parser import (
    parse_event_conversation,
)

from src.markets.id_market import (
    load_id_vwap_for_timestamps,
)

from src.optimization.da_event_optimizer import (
    optimize_da_with_events,
)

from src.optimization.id_optimizer import (
    optimize_intraday,
)


TIMEZONE = "Europe/Stockholm"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

DA_SCHEDULE_DIR = (
    DATA_DIR
    / "output_data"
    / "schedules"
    / "da"
)

RESCHEDULED_ROOT = (
    DATA_DIR
    / "output_data"
    / "schedules"
    / "rescheduled"
)

EVENT_OUTPUT_DIR = (
    DATA_DIR
    / "scenarios"
    / "events"
)

ID_PRICE_PATH = (
    DATA_DIR
    / "input_data"
    / "id_price"
    / "id_price.xlsx"
)

BESS_PATH = (
    DATA_DIR
    / "parameters"
    / "bess.json"
)

MARKET_PATH = (
    DATA_DIR
    / "parameters"
    / "market.json"
)

EVENT_POLICY_PATH = (
    DATA_DIR
    / "parameters"
    / "event_policy.json"
)

LOGISTICS_LIBRARY_PATH = (
    DATA_DIR
    / "parameters"
    / "logistic_events.json"
)

MAX_CLARIFICATION_TURNS = 5


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stateful operational event agent with "
            "ADD / CANCEL / UPDATE and DA/ID routing."
        )
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
        "--message",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--event-file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--reset-events",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# MARKET STAGE
# ============================================================

def parse_stockholm_timestamp(
    value,
):
    ts = pd.Timestamp(
        value
    )

    if ts.tzinfo is None:

        return ts.tz_localize(
            TIMEZONE
        )

    return ts.tz_convert(
        TIMEZONE
    )


def get_da_gate_closure(
    delivery_day,
    market,
):
    gate_time = market.get(
        "da_gate_closure_local_time",
        "12:00",
    )

    hour_text, minute_text = (
        gate_time.split(
            ":"
        )
    )

    delivery_date = pd.Timestamp(
        delivery_day
    )

    gate_date = (
        delivery_date
        -
        pd.Timedelta(
            days=1
        )
    )

    gate_naive = pd.Timestamp(
        year=gate_date.year,
        month=gate_date.month,
        day=gate_date.day,
        hour=int(
            hour_text
        ),
        minute=int(
            minute_text
        ),
    )

    return gate_naive.tz_localize(
        TIMEZONE
    )


def determine_market_stage(
    delivery_day,
    as_of,
    market,
):
    as_of_ts = parse_stockholm_timestamp(
        as_of
    )

    gate_closure = get_da_gate_closure(
        delivery_day=delivery_day,
        market=market,
    )

    stage = (
        "DA"
        if as_of_ts < gate_closure
        else "ID"
    )

    return (
        stage,
        as_of_ts,
        gate_closure,
    )


# ============================================================
# BASELINE SCHEDULES
# ============================================================

def load_original_da_schedule(
    day,
):
    path = (
        DA_SCHEDULE_DIR
        /
        f"da_schedule_{day}.xlsx"
    )

    if not path.exists():

        raise FileNotFoundError(
            "DA schedule not found. "
            "Run run_da.py first:\n"
            f"{path}"
        )

    return (
        pd.read_excel(
            path,
            sheet_name="schedule",
        ),
        path,
    )


def load_da_commitment_for_id(
    day,
):
    """
    Prefer the current pre-gate DA reschedule as the market
    commitment. Otherwise use the original DA schedule.
    """

    path = (
        RESCHEDULED_ROOT
        / "da"
        / f"da_rescheduled_{day}.xlsx"
    )

    if path.exists():

        print(
            "\nUsing current pre-gate DA reschedule "
            "as market commitment:"
        )

        print(
            f"  {path}"
        )

        return (
            pd.read_excel(
                path,
                sheet_name="schedule",
            ),
            path,
        )

    return load_original_da_schedule(
        day
    )




def load_current_id_history(
    day,
    as_of,
):
    """
    Load the current ID-rescheduled schedule only when it comes
    from the same or an earlier simulation time.

    This preserves executed history while preventing future leakage
    if backtest commands are run out of chronological order.
    """

    path = (
        RESCHEDULED_ROOT
        / "id"
        / f"id_rescheduled_{day}.xlsx"
    )

    if not path.exists():

        return (
            None,
            None,
        )

    try:
        summary_df = pd.read_excel(
            path,
            sheet_name="summary",
        )

        summary = dict(
            zip(
                summary_df["metric"],
                summary_df["value"],
            )
        )

        previous_as_of_raw = summary.get(
            "as_of"
        )

        if previous_as_of_raw is None:

            print(
                "\nExisting ID schedule has no as-of metadata; "
                "not using it as history."
            )

            return (
                None,
                None,
            )

        previous_as_of = parse_stockholm_timestamp(
            previous_as_of_raw
        )

        current_as_of = parse_stockholm_timestamp(
            as_of
        )

        if previous_as_of > current_as_of:

            print(
                "\nExisting ID schedule is from a later simulation "
                "time, so it is NOT used as history."
            )

            return (
                None,
                None,
            )

        print(
            "\nUsing current ID-rescheduled schedule "
            "as executed-history baseline:"
        )

        print(
            f"  {path}"
        )

        return (
            pd.read_excel(
                path,
                sheet_name="schedule",
            ),
            path,
        )

    except Exception as exc:

        print(
            "\nCould not use existing ID schedule as history: "
            f"{exc}"
        )

        return (
            None,
            None,
        )


# ============================================================
# OPERATION NORMALIZATION
# ============================================================

def _normalize_one_event(
    raw_event,
    day,
    as_of,
    event_policy,
):
    """
    Reuse the existing deterministic event_handler but keep only
    the event explicitly supplied by the current operation.

    This prevents automatically scheduled logistics library
    entries from being accidentally inserted into the ledger
    during every ADD/UPDATE operation.
    """

    normalized = normalize_events(
        parsed_payload={
            "events": [
                raw_event
            ]
        },
        day=day,
        as_of=as_of,
        event_policy=event_policy,
        logistics_library_path=LOGISTICS_LIBRARY_PATH,
    )

    explicit_events = [
        event
        for event
        in normalized.get(
            "events",
            [],
        )
        if event.get(
            "source"
        )
        ==
        "llm_event"
    ]

    if len(
        explicit_events
    ) != 1:

        raise ValueError(
            "Expected exactly one normalized explicit event, "
            f"received {len(explicit_events)}."
        )

    return explicit_events[0]


def normalize_operations(
    parsed_payload,
    day,
    as_of,
    event_policy,
):
    """
    Physical defaults are added only to ADD events and UPDATE
    replacement events.

    CANCEL requires only an existing event_id.
    """

    raw_operations = parsed_payload.get(
        "operations"
    )

    # Backward-compatible deterministic event files:
    # {"events": [...]} means ADD.
    if raw_operations is None:

        legacy_events = parsed_payload.get(
            "events",
            [],
        )

        raw_operations = [
            {
                "action":
                    "add",

                "event":
                    event,
            }
            for event
            in legacy_events
        ]

    normalized_operations = []

    for operation in raw_operations:

        action = str(
            operation[
                "action"
            ]
        ).lower()

        if action == "add":

            normalized_operations.append(
                {
                    "action":
                        "add",

                    "event":
                        _normalize_one_event(
                            raw_event=operation[
                                "event"
                            ],
                            day=day,
                            as_of=as_of,
                            event_policy=event_policy,
                        ),
                }
            )

        elif action == "cancel":

            normalized_operations.append(
                {
                    "action":
                        "cancel",

                    "event_id":
                        operation[
                            "event_id"
                        ],
                }
            )

        elif action == "update":

            normalized_operations.append(
                {
                    "action":
                        "update",

                    "event_id":
                        operation[
                            "event_id"
                        ],

                    "replacement_event":
                        _normalize_one_event(
                            raw_event=operation[
                                "replacement_event"
                            ],
                            day=day,
                            as_of=as_of,
                            event_policy=event_policy,
                        ),
                }
            )

        else:

            raise ValueError(
                "Unsupported operation: "
                f"{action}"
            )

    return normalized_operations


# ============================================================
# HUMAN-IN-THE-LOOP EVENT SELECTION
# ============================================================

def _event_time_window_text(
    event,
):
    if (
        event.get(
            "event_type"
        )
        ==
        "company_ev_charging"
    ):

        start = event.get(
            "arrival_time"
        )

        end = event.get(
            "departure_time"
        )

    else:

        start = event.get(
            "start_time"
        )

        end = event.get(
            "end_time"
        )

    def _clock(value):

        if not value:
            return "?"

        try:

            return (
                parse_stockholm_timestamp(
                    value
                )
                .strftime(
                    "%H:%M"
                )
            )

        except Exception:

            return str(
                value
            )

    return (
        f"{_clock(start)}-"
        f"{_clock(end)}"
    )


def _event_details_text(
    event,
):
    event_type = event.get(
        "event_type"
    )

    if (
        event_type
        ==
        "company_ev_charging"
    ):

        return (
            f"{event.get('vehicles', '?')} vehicle(s)"
        )

    if (
        event_type
        ==
        "logistics_load"
    ):

        process = event.get(
            "logistic_event_name",
            "logistics_load",
        )

        power = event.get(
            "power_kw"
        )

        if power is None:

            return str(
                process
            )

        return (
            f"{process}, "
            f"{float(power):g} kW"
        )

    return ""


def print_active_event_selection(
    known_events,
):
    print(
        "\nAgent > Select the active event:"
    )

    print(
        "  #  EVENT ID       TYPE                  TIME         DETAILS"
    )

    print(
        "  -  -------------  --------------------  -----------  ------------------------------"
    )

    for index, event in enumerate(
        known_events,
        start=1,
    ):

        print(
            f"  {index:<2} "
            f"{str(event.get('event_id', '')):<13}  "
            f"{str(event.get('event_type', '')):<20}  "
            f"{_event_time_window_text(event):<11}  "
            f"{_event_details_text(event)}"
        )


def _event_id_explicitly_in_message(
    message,
    known_events,
):
    message_lower = str(
        message
    ).lower()

    matches = [
        event
        for event
        in known_events
        if (
            event.get(
                "event_id"
            )
            and
            str(
                event[
                    "event_id"
                ]
            ).lower()
            in message_lower
        )
    ]

    if len(
        matches
    ) == 1:

        return matches[0]

    return None


def select_active_event(
    known_events,
    original_message=None,
):
    """
    Deterministic human-in-the-loop selection.

    Accept:
        1
        2
        ...
        exact event_id

    If the original message already contains exactly one active
    event_id, no additional prompt is needed.
    """

    if not known_events:

        print(
            "\nAgent > There are no active events to select."
        )

        return None

    explicit = _event_id_explicitly_in_message(
        message=original_message
        or "",
        known_events=known_events,
    )

    if explicit is not None:

        print(
            "\nAgent > Using the event ID explicitly provided "
            "in the instruction:"
        )

        print(
            f"  {explicit['event_id']}"
        )

        return explicit

    print_active_event_selection(
        known_events
    )

    while True:

        selection = input(
            "\nStakeholder [event # or ID] > "
        ).strip()

        if not selection:

            print(
                "\nNo event selected. "
                "No rescheduling performed."
            )

            return None

        if selection.isdigit():

            index = int(
                selection
            )

            if (
                1
                <= index
                <= len(
                    known_events
                )
            ):

                return known_events[
                    index - 1
                ]

        for event in known_events:

            if (
                str(
                    event.get(
                        "event_id",
                        "",
                    )
                )
                ==
                selection
            ):

                return event

        print(
            "Agent > Invalid selection. "
            "Enter a list number or exact event ID."
        )


def semantic_event_for_update(
    event,
):
    """
    Strip ledger/audit/physical metadata and keep only the semantic
    fields needed to reconstruct the event deterministically.
    """

    event_type = event.get(
        "event_type"
    )

    if (
        event_type
        ==
        "company_ev_charging"
    ):

        return {
            "event_type":
                "company_ev_charging",

            "vehicles":
                int(
                    event[
                        "vehicles"
                    ]
                ),

            "arrival_time":
                event[
                    "arrival_time"
                ],

            "departure_time":
                event[
                    "departure_time"
                ],
        }

    if (
        event_type
        ==
        "logistics_load"
    ):

        semantic = {
            "event_type":
                "logistics_load",

            "start_time":
                event[
                    "start_time"
                ],

            "end_time":
                event[
                    "end_time"
                ],
        }

        if event.get(
            "logistic_event_name"
        ):

            semantic[
                "logistic_event_name"
            ] = event[
                "logistic_event_name"
            ]

        elif event.get(
            "power_kw"
        ) is not None:

            semantic[
                "power_kw"
            ] = float(
                event[
                    "power_kw"
                ]
            )

        return semantic

    raise ValueError(
        "Unsupported selected event type: "
        f"{event_type}"
    )


def build_update_replacement(
    selected_event,
    changes,
):
    replacement = semantic_event_for_update(
        selected_event
    )

    event_type = replacement[
        "event_type"
    ]

    if (
        event_type
        ==
        "company_ev_charging"
    ):

        allowed = {
            "vehicles",
            "arrival_time",
            "departure_time",
        }

    elif (
        event_type
        ==
        "logistics_load"
    ):

        allowed = {
            "logistic_event_name",
            "start_time",
            "end_time",
            "power_kw",
        }

    else:

        raise ValueError(
            f"Unsupported event type: {event_type}"
        )

    unsupported = (
        set(
            changes.keys()
        )
        -
        allowed
    )

    if unsupported:

        raise ValueError(
            "The requested update contains fields that do not "
            f"belong to the selected {event_type} event: "
            f"{sorted(unsupported)}"
        )

    replacement.update(
        changes
    )

    return replacement



def collect_selected_event_update_changes(
    selected_event,
    day,
    as_of,
    logistics_catalog,
):
    """
    After the stakeholder selects the exact event, collect ONLY
    the requested semantic changes.

    This keeps the conversational order intuitive:

        UPDATE intent
        -> select event
        -> ask what should change
        -> parse changes
    """

    print(
        "\nAgent > Selected event:"
    )

    print(
        f"  {selected_event['event_id']} | "
        f"{selected_event['event_type']} | "
        f"{_event_time_window_text(selected_event)} | "
        f"{_event_details_text(selected_event)}"
    )

    print(
        "\nAgent > What should be changed in this event?"
    )

    conversation = [
        {
            "role":
                "system",

            "content":
                (
                    "The stakeholder has already selected exactly one "
                    "active event for UPDATE. Do not select another event. "
                    "Interpret the stakeholder's next message only as the "
                    "requested semantic changes to this selected event. "
                    "Return action='update' and a non-empty 'changes' object."
                ),
        }
    ]

    for _ in range(
        MAX_CLARIFICATION_TURNS
    ):

        user_message = input(
            "Stakeholder > "
        ).strip()

        if not user_message:

            print(
                "\nNo update details received. "
                "No rescheduling performed."
            )

            return None

        conversation.append(
            {
                "role":
                    "user",

                "content":
                    user_message,
            }
        )

        parsed = parse_event_conversation(
            conversation=conversation,
            day=day,
            as_of=as_of,
            known_events=[
                selected_event
            ],
            logistics_catalog=logistics_catalog,
        )

        if (
            parsed.get(
                "status"
            )
            ==
            "actionable"
        ):

            update_operations = [
                operation
                for operation
                in parsed.get(
                    "operations",
                    [],
                )
                if (
                    operation.get(
                        "action"
                    )
                    ==
                    "update"
                )
            ]

            if update_operations:

                changes = update_operations[
                    0
                ].get(
                    "changes",
                    {},
                )

                if changes:

                    return changes

        if (
            parsed.get(
                "status"
            )
            ==
            "needs_clarification"
        ):

            question = (
                parsed.get(
                    "question"
                )
                or
                "Please specify the requested change."
            )

        else:

            question = (
                "Please specify what should change "
                "in the selected event."
            )

        print(
            f"\nAgent > {question}"
        )

        conversation.append(
            {
                "role":
                    "assistant",

                "content":
                    question,
            }
        )

    print(
        "\nThe requested update could not be made actionable "
        "within the clarification limit."
    )

    return None


def resolve_human_selected_operations(
    parsed_payload,
    known_events,
    original_message,
    day,
    as_of,
    logistics_catalog,
):
    """
    Convert LLM intent into the existing deterministic ledger
    operation format.

    LLM:
        ADD    -> complete event
        CANCEL -> intent only
        UPDATE -> patch only

    Python + stakeholder:
        choose exact event
        produce event_id
        produce complete replacement_event
    """

    resolved = []

    for operation in parsed_payload.get(
        "operations",
        [],
    ):

        action = operation[
            "action"
        ]

        if action == "add":

            resolved.append(
                operation
            )

            continue

        selected_event = select_active_event(
            known_events=known_events,
            original_message=original_message,
        )

        if selected_event is None:

            return None

        event_id = selected_event[
            "event_id"
        ]

        if action == "cancel":

            resolved.append(
                {
                    "action":
                        "cancel",

                    "event_id":
                        event_id,
                }
            )

        elif action == "update":

            changes = operation.get(
                "changes",
                {},
            )

            if not changes:

                changes = (
                    collect_selected_event_update_changes(
                        selected_event=selected_event,
                        day=day,
                        as_of=as_of,
                        logistics_catalog=logistics_catalog,
                    )
                )

                if changes is None:

                    return None

            replacement = (
                build_update_replacement(
                    selected_event=selected_event,
                    changes=changes,
                )
            )

            resolved.append(
                {
                    "action":
                        "update",

                    "event_id":
                        event_id,

                    "replacement_event":
                        replacement,
                }
            )

        else:

            raise ValueError(
                f"Unsupported operation: {action}"
            )

    result = dict(
        parsed_payload
    )

    result[
        "operations"
    ] = resolved

    return result


# ============================================================
# CONVERSATIONAL INTAKE
# ============================================================

def collect_actionable_operations(
    day,
    as_of,
    known_events,
    logistics_catalog,
    first_message=None,
):
    conversation = []

    if first_message is None:

        first_message = input(
            "\nStakeholder > "
        ).strip()

    if not first_message:

        print(
            "\nNo message received. "
            "No rescheduling performed."
        )

        return None

    user_message = first_message

    for _ in range(
        MAX_CLARIFICATION_TURNS
    ):

        conversation.append(
            {
                "role":
                    "user",

                "content":
                    user_message,
            }
        )

        parsed = parse_event_conversation(
            conversation=conversation,
            day=day,
            as_of=as_of,
            known_events=known_events,
            logistics_catalog=logistics_catalog,
        )

        status = parsed[
            "status"
        ]

        operations = parsed.get(
            "operations",
            [],
        )

        if (
            status == "actionable"
            and operations
        ):

            print(
                "\nAgent > Operational instruction understood."
            )

            for operation in operations:

                print(
                    f"  - {operation['action'].upper()}"
                )

            needs_selection = any(
                operation[
                    "action"
                ]
                in {
                    "cancel",
                    "update",
                }
                for operation
                in operations
            )

            if needs_selection:

                resolved = resolve_human_selected_operations(
                    parsed_payload=parsed,
                    known_events=known_events,
                    original_message=conversation[
                        0
                    ][
                        "content"
                    ],
                    day=day,
                    as_of=as_of,
                    logistics_catalog=logistics_catalog,
                )

                if resolved is None:

                    return None

                return resolved

            return parsed

        if status == "no_action":

            print(
                "\nAgent > No operational change detected. "
                "No rescheduling is required."
            )

            return None

        question = (
            parsed.get(
                "question"
            )
            or
            "Please provide the missing operational details."
        )

        print(
            f"\nAgent > {question}"
        )

        conversation.append(
            {
                "role":
                    "assistant",

                "content":
                    question,
            }
        )

        user_message = input(
            "Stakeholder > "
        ).strip()

        if not user_message:

            print(
                "\nNo follow-up received. "
                "No rescheduling performed."
            )

            return None

    print(
        "\nInstruction could not be made actionable "
        "within the clarification limit."
    )

    return None


# ============================================================
# OUTPUT
# ============================================================

def save_outputs(
    result,
    summary,
    optimizer_events,
    operation_results,
    ledger_path,
    day,
    as_of,
    market_stage,
    gate_closure,
    baseline_path,
):
    stage_lower = market_stage.lower()

    schedule_dir = (
        RESCHEDULED_ROOT
        / stage_lower
    )

    schedule_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVENT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if market_stage == "DA":

        schedule_name = (
            f"da_rescheduled_{day}.xlsx"
        )

    else:

        schedule_name = (
            f"id_rescheduled_{day}.xlsx"
        )

    schedule_path = (
        schedule_dir
        / schedule_name
    )

    event_snapshot_path = (
        EVENT_OUTPUT_DIR
        /
        f"current_optimizer_events_{day}.json"
    )

    excel_result = result.copy()

    for column in [
        "timestamp",
        "timestamp_utc",
    ]:

        if column in excel_result.columns:

            excel_result[
                column
            ] = (
                excel_result[
                    column
                ]
                .astype(
                    str
                )
            )

    summary = dict(
        summary
    )

    summary[
        "market_stage"
    ] = market_stage

    summary[
        "da_gate_closure"
    ] = gate_closure.isoformat()

    summary[
        "baseline_schedule"
    ] = str(
        baseline_path
    )

    summary[
        "event_ledger"
    ] = str(
        ledger_path
    )

    summary_df = pd.DataFrame(
        {
            "metric":
                list(
                    summary.keys()
                ),

            "value":
                list(
                    summary.values()
                ),
        }
    )

    with pd.ExcelWriter(
        schedule_path,
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

    snapshot = {
        "day":
            day,

        "as_of":
            as_of,

        "market_stage":
            market_stage,

        "operation_results":
            operation_results,

        "optimizer_events":
            optimizer_events,

        "event_ledger":
            str(
                ledger_path
            ),

        "baseline_schedule":
            str(
                baseline_path
            ),
    }

    with open(
        event_snapshot_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            snapshot,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return (
        schedule_path,
        event_snapshot_path,
        summary,
    )


def print_operation_results(
    operation_results,
):
    print(
        "\nLedger operations:"
    )

    for item in operation_results:

        action = item[
            "action"
        ].upper()

        result = item[
            "result"
        ]

        if action == "UPDATE":

            print(
                f"  - UPDATE {item['event_id']} "
                f"-> {item.get('replacement_event_id')} "
                f"({result})"
            )

        else:

            print(
                f"  - {action} "
                f"{item.get('event_id')} "
                f"({result})"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if (
        args.message
        and
        args.event_file
    ):

        raise ValueError(
            "Use either --message or --event-file, not both."
        )

    print(
        "\n"
        "============================================================\n"
        "OPERATIONAL EVENT / STATEFUL MARKET-AWARE AGENT\n"
        "============================================================"
    )

    print(
        f"\nDelivery day: {args.day}"
    )

    print(
        f"As-of:        {args.as_of}"
    )

    market = load_json(
        MARKET_PATH
    )

    bess = load_json(
        BESS_PATH
    )

    event_policy = load_json(
        EVENT_POLICY_PATH
    )

    logistics_library = load_logistics_library(
        LOGISTICS_LIBRARY_PATH
    )

    logistics_catalog = logistics_catalog_for_llm(
        logistics_library
    )

    if logistics_catalog:

        print(
            "\nOperational logistics catalog:"
        )

        for item in logistics_catalog:

            unit_text = (
                f" | unit: {item['unit_name']}"
                if item.get(
                    "unit_name"
                )
                else ""
            )

            print(
                f"  - {item['name']}{unit_text}"
            )

    market_stage, as_of_ts, gate_closure = (
        determine_market_stage(
            delivery_day=args.day,
            as_of=args.as_of,
            market=market,
        )
    )

    print(
        f"\nDA gate closure: "
        f"{gate_closure.isoformat()}"
    )

    print(
        "Market stage:    "
        +
        (
            "PRE-GATE / DAY-AHEAD"
            if market_stage == "DA"
            else
            "POST-GATE / INTRADAY"
        )
    )

    # ========================================================
    # 0. LOAD LEDGER
    # ========================================================

    if args.reset_events:

        reset_event_ledger(
            event_directory=EVENT_OUTPUT_DIR,
            day=args.day,
        )

        print(
            "\nEvent ledger reset."
        )

    ledger = load_event_ledger(
        event_directory=EVENT_OUTPUT_DIR,
        day=args.day,
    )

    known_events = active_events(
        ledger=ledger,
        as_of=args.as_of,
    )

    print(
        f"\nCurrently active known events: "
        f"{len(known_events)}"
    )

    for event in known_events:

        print(
            f"  - {event['event_id']} "
            f"{event['event_type']}"
        )

    # ========================================================
    # 1. PARSE OPERATION(S)
    # ========================================================

    if args.event_file:

        event_file = Path(
            args.event_file
        )

        if not event_file.is_absolute():

            event_file = (
                PROJECT_ROOT
                / event_file
            )

        with open(
            event_file,
            "r",
            encoding="utf-8",
        ) as file:

            parsed_payload = json.load(
                file
            )

    else:

        print(
            "\nStarting conversational event intake..."
        )

        parsed_payload = collect_actionable_operations(
            day=args.day,
            as_of=args.as_of,
            known_events=known_events,
            logistics_catalog=logistics_catalog,
            first_message=args.message,
        )

        if parsed_payload is None:
            return

    normalized_operations = normalize_operations(
        parsed_payload=parsed_payload,
        day=args.day,
        as_of=args.as_of,
        event_policy=event_policy,
    )

    if not normalized_operations:

        print(
            "\nNo operational changes to apply."
        )

        return

    # ========================================================
    # 2. APPLY TO AUDIT LEDGER
    # ========================================================

    (
        ledger,
        operation_results,
        changed,
    ) = apply_operations(
        ledger=ledger,
        operations=normalized_operations,
        as_of=args.as_of,
        market_stage=market_stage,
    )

    ledger_path = save_event_ledger(
        event_directory=EVENT_OUTPUT_DIR,
        ledger=ledger,
    )

    print_operation_results(
        operation_results
    )

    if not changed:

        print(
            "\nOperational state did not change. "
            "No rescheduling is required."
        )

        return

    # ========================================================
    # 3. CURRENT ACTIVE PHYSICAL STATE
    # ========================================================

    optimizer_events = events_for_optimizer(
        ledger=ledger,
        as_of=args.as_of,
        market_stage=market_stage,
    )

    print(
        f"\nActive events used by {market_stage} optimizer: "
        f"{len(optimizer_events)}"
    )

    for event in optimizer_events:

        print(
            f"  - {event['event_id']} "
            f"{event['event_type']}"
        )

    combined_events = {
        "day":
            args.day,

        "as_of":
            args.as_of,

        "events":
            optimizer_events,
    }

    # IMPORTANT:
    # zero active events is still a valid optimization state.
    #
    # Example:
    # cancel the only pre-gate event.
    #
    # DA should revert toward baseline.
    # ID should trade away the now-unneeded DA commitment.

    # ========================================================
    # 4A. DA
    # ========================================================

    if market_stage == "DA":

        da_schedule, baseline_path = (
            load_original_da_schedule(
                args.day
            )
        )

        print(
            "\nRouting CURRENT active event state "
            "to DAY-AHEAD optimizer."
        )

        result, summary = optimize_da_with_events(
            da_schedule=da_schedule,
            normalized_events=combined_events,
            bess=bess,
            market=market,
            event_policy=event_policy,
        )

    # ========================================================
    # 4B. ID
    # ========================================================

    else:

        da_schedule, baseline_path = (
            load_da_commitment_for_id(
                args.day
            )
        )

        history_schedule, history_path = (
            load_current_id_history(
                day=args.day,
                as_of=args.as_of,
            )
        )

        print(
            "\nRouting CURRENT active physical event state "
            "to INTRADAY optimizer."
        )

        print(
            "DA grid profile remains the market commitment."
        )

        timestamps_utc = pd.to_datetime(
            da_schedule[
                "timestamp_utc"
            ],
            utc=True,
        )

        id_prices = load_id_vwap_for_timestamps(
            path=ID_PRICE_PATH,
            target_timestamps_utc=timestamps_utc,
        )

        result, summary = optimize_intraday(
            da_schedule=da_schedule,
            id_prices_eur_mwh=id_prices,
            normalized_events=combined_events,
            as_of=args.as_of,
            bess=bess,
            market=market,
            event_policy=event_policy,
            history_schedule=history_schedule,
        )

        if history_path is not None:

            summary[
                "history_schedule"
            ] = str(
                history_path
            )

    # ========================================================
    # 5. SAVE CURRENT SCHEDULE
    # ========================================================

    (
        schedule_path,
        event_snapshot_path,
        summary,
    ) = save_outputs(
        result=result,
        summary=summary,
        optimizer_events=optimizer_events,
        operation_results=operation_results,
        ledger_path=ledger_path,
        day=args.day,
        as_of=args.as_of,
        market_stage=market_stage,
        gate_closure=gate_closure,
        baseline_path=baseline_path,
    )

    # ========================================================
    # 6. PLOT
    # ========================================================

    print(
        "\n"
        "============================================================\n"
        "RESCHEDULED VISUALIZATION\n"
        "============================================================"
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.visualization.plot_rescheduled",
            "--schedule-file",
            str(
                schedule_path
            ),
            "--day",
            args.day,
            "--as-of",
            args.as_of,
            "--stage",
            market_stage,
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    # ========================================================
    # 7. RESULT
    # ========================================================

    print(
        "\n"
        "============================================================\n"
        f"{market_stage} RESCHEDULING RESULT\n"
        "============================================================"
    )

    print(
        f"\nCompany EV requested: "
        f"{float(summary.get('company_ev_requested_kwh', 0.0)):.1f} kWh"
    )

    print(
        f"Company EV delivered: "
        f"{float(summary.get('company_ev_delivered_kwh', 0.0)):.1f} kWh"
    )

    print(
        f"Company EV unserved:  "
        f"{float(summary.get('company_ev_unserved_kwh', 0.0)):.1f} kWh"
    )

    print(
        f"Critical logistics unserved: "
        f"{float(summary.get('critical_logistics_unserved_kwh', 0.0)):.1f} kWh"
    )

    if market_stage == "DA":

        print(
            f"DAM energy cost: "
            f"{float(summary.get('dam_energy_cost_eur', 0.0)):.2f} EUR"
        )

    else:

        print(
            f"ID trade cost: "
            f"{float(summary.get('id_trade_cost_eur', 0.0)):.2f} EUR"
        )

    print(
        f"\nCurrent schedule:\n"
        f"  {schedule_path}"
    )

    print(
        f"\nPersistent audit ledger:\n"
        f"  {ledger_path}"
    )

    print(
        f"\nCurrent optimizer event snapshot:\n"
        f"  {event_snapshot_path}"
    )


if __name__ == "__main__":
    main()
