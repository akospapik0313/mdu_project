from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd


TIMEZONE = "Europe/Stockholm"


def _to_stockholm(
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


def _canonical_timestamp(
    value,
):
    return _to_stockholm(
        value
    ).isoformat()


def _event_identity(
    event,
):
    event_type = event[
        "event_type"
    ]

    if (
        event_type
        ==
        "company_ev_charging"
    ):

        return {
            "event_type":
                event_type,

            "vehicles":
                int(
                    event[
                        "vehicles"
                    ]
                ),

            "arrival_time":
                _canonical_timestamp(
                    event[
                        "arrival_time"
                    ]
                ),

            "departure_time":
                _canonical_timestamp(
                    event[
                        "departure_time"
                    ]
                ),
        }

    if (
        event_type
        ==
        "logistics_load"
    ):

        identity = {
            "event_type":
                event_type,

            "start_time":
                _canonical_timestamp(
                    event[
                        "start_time"
                    ]
                ),

            "end_time":
                _canonical_timestamp(
                    event[
                        "end_time"
                    ]
                ),
        }

        if (
            event.get(
                "power_kw"
            )
            is not None
        ):

            identity[
                "power_kw"
            ] = float(
                event[
                    "power_kw"
                ]
            )

        if event.get(
            "logistic_event_name"
        ):

            identity[
                "logistic_event_name"
            ] = str(
                event[
                    "logistic_event_name"
                ]
            )

        return identity

    raise ValueError(
        "Unsupported event type: "
        f"{event_type}"
    )


def _identity_key(
    event,
):
    return json.dumps(
        _event_identity(
            event
        ),
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def _base_event_id(
    event,
):
    return hashlib.sha1(
        _identity_key(
            event
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:12]


def _unique_event_id(
    event,
    existing_ids,
):
    base = _base_event_id(
        event
    )

    if base not in existing_ids:

        return base

    counter = 2

    while True:

        candidate = (
            f"{base}-{counter}"
        )

        if candidate not in existing_ids:

            return candidate

        counter += 1


def ledger_path(
    event_directory,
    day,
):
    return (
        Path(
            event_directory
        )
        /
        f"event_ledger_{day}.json"
    )


def _migrate_ledger(
    ledger,
):
    """
    Existing ledgers from the previous prototype remain valid.

    Missing status means active.
    Missing event_id is generated.
    """

    result = deepcopy(
        ledger
    )

    result.setdefault(
        "events",
        [],
    )

    existing_ids = {
        event.get(
            "event_id"
        )
        for event
        in result[
            "events"
        ]
        if event.get(
            "event_id"
        )
    }

    for event in result[
        "events"
    ]:

        event.setdefault(
            "status",
            "active",
        )

        if not event.get(
            "event_id"
        ):

            identifier = _unique_event_id(
                event,
                existing_ids,
            )

            event[
                "event_id"
            ] = identifier

            existing_ids.add(
                identifier
            )

    return result


def load_event_ledger(
    event_directory,
    day,
):
    path = ledger_path(
        event_directory=event_directory,
        day=day,
    )

    if not path.exists():

        return {
            "day":
                day,

            "events":
                [],
        }

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        ledger = json.load(
            file
        )

    if ledger.get(
        "day"
    ) != day:

        raise ValueError(
            "Event ledger delivery day mismatch."
        )

    return _migrate_ledger(
        ledger
    )


def save_event_ledger(
    event_directory,
    ledger,
):
    event_directory = Path(
        event_directory
    )

    event_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = ledger_path(
        event_directory=event_directory,
        day=ledger[
            "day"
        ],
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            ledger,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return path


def reset_event_ledger(
    event_directory,
    day,
):
    path = ledger_path(
        event_directory=event_directory,
        day=day,
    )

    if path.exists():

        path.unlink()

        return True

    return False


def _event_end_time(
    event,
):
    if (
        event[
            "event_type"
        ]
        ==
        "company_ev_charging"
    ):

        return _to_stockholm(
            event[
                "departure_time"
            ]
        )

    if (
        event[
            "event_type"
        ]
        ==
        "logistics_load"
    ):

        return _to_stockholm(
            event[
                "end_time"
            ]
        )

    raise ValueError(
        "Unsupported event type: "
        f"{event['event_type']}"
    )


def active_events(
    ledger,
    as_of=None,
):
    """
    Active operational state.

    If as_of is supplied, already-ended events are omitted from
    the operational view, although they remain in the audit ledger.
    """

    as_of_ts = (
        _to_stockholm(
            as_of
        )
        if as_of is not None
        else None
    )

    selected = []

    for event in ledger.get(
        "events",
        [],
    ):

        if (
            event.get(
                "status",
                "active",
            )
            !=
            "active"
        ):

            continue

        if (
            as_of_ts is not None
            and
            _event_end_time(
                event
            )
            <= as_of_ts
        ):

            continue

        selected.append(
            deepcopy(
                event
            )
        )

    return selected


def events_for_optimizer(
    ledger,
    as_of,
    market_stage,
):
    """
    Both DA and ID optimization receive the complete CURRENT
    active physical event state.

    DA:
        original forecast baseline + active events.

    ID:
        DA grid schedule is the MARKET commitment,
        while physical demand is rebuilt from:
            base load forecast + current active events.

    This is what makes post-gate cancellation/update possible.
    """

    return active_events(
        ledger=ledger,
        as_of=as_of,
    )


def _find_active_event(
    ledger,
    event_id,
):
    for event in ledger.get(
        "events",
        [],
    ):

        if (
            event.get(
                "event_id"
            )
            ==
            event_id
            and
            event.get(
                "status",
                "active",
            )
            ==
            "active"
        ):

            return event

    return None


def _find_active_duplicate(
    ledger,
    candidate_event,
):
    wanted = _identity_key(
        candidate_event
    )

    for event in ledger.get(
        "events",
        [],
    ):

        if (
            event.get(
                "status",
                "active",
            )
            !=
            "active"
        ):

            continue

        if (
            _identity_key(
                event
            )
            ==
            wanted
        ):

            return event

    return None


def apply_operations(
    ledger,
    operations,
    as_of,
    market_stage,
):
    """
    Apply normalized ADD / CANCEL / UPDATE operations.

    CANCEL:
        event remains in ledger with status=cancelled.

    UPDATE:
        old event becomes status=superseded;
        replacement becomes a new active event.

    Returns:
        updated_ledger,
        operation_results,
        changed
    """

    result = _migrate_ledger(
        ledger
    )

    timestamp = _to_stockholm(
        as_of
    ).isoformat()

    operation_results = []
    changed = False

    for operation in operations:

        action = operation[
            "action"
        ]

        # ====================================================
        # ADD
        # ====================================================

        if action == "add":

            candidate = deepcopy(
                operation[
                    "event"
                ]
            )

            duplicate = _find_active_duplicate(
                result,
                candidate,
            )

            if duplicate is not None:

                duplicate[
                    "last_seen_as_of"
                ] = timestamp

                operation_results.append(
                    {
                        "action":
                            "add",

                        "result":
                            "duplicate_ignored",

                        "event_id":
                            duplicate[
                                "event_id"
                            ],
                    }
                )

                continue

            existing_ids = {
                event.get(
                    "event_id"
                )
                for event
                in result[
                    "events"
                ]
                if event.get(
                    "event_id"
                )
            }

            identifier = _unique_event_id(
                candidate,
                existing_ids,
            )

            candidate.update(
                {
                    "event_id":
                        identifier,

                    "status":
                        "active",

                    "first_seen_as_of":
                        timestamp,

                    "last_seen_as_of":
                        timestamp,

                    "first_seen_market_stage":
                        market_stage,
                }
            )

            result[
                "events"
            ].append(
                candidate
            )

            operation_results.append(
                {
                    "action":
                        "add",

                    "result":
                        "added",

                    "event_id":
                        identifier,
                }
            )

            changed = True

        # ====================================================
        # CANCEL
        # ====================================================

        elif action == "cancel":

            target_id = operation[
                "event_id"
            ]

            target = _find_active_event(
                result,
                target_id,
            )

            if target is None:

                raise ValueError(
                    "Cannot cancel event because it is not "
                    f"active: {target_id}"
                )

            target[
                "status"
            ] = "cancelled"

            target[
                "cancelled_as_of"
            ] = timestamp

            target[
                "cancelled_market_stage"
            ] = market_stage

            target[
                "last_updated_as_of"
            ] = timestamp

            operation_results.append(
                {
                    "action":
                        "cancel",

                    "result":
                        "cancelled",

                    "event_id":
                        target_id,
                }
            )

            changed = True

        # ====================================================
        # UPDATE
        # ====================================================

        elif action == "update":

            target_id = operation[
                "event_id"
            ]

            target = _find_active_event(
                result,
                target_id,
            )

            if target is None:

                raise ValueError(
                    "Cannot update event because it is not "
                    f"active: {target_id}"
                )

            replacement = deepcopy(
                operation[
                    "replacement_event"
                ]
            )

            # No semantic change -> no-op.
            if (
                _identity_key(
                    target
                )
                ==
                _identity_key(
                    replacement
                )
            ):

                target[
                    "last_seen_as_of"
                ] = timestamp

                operation_results.append(
                    {
                        "action":
                            "update",

                        "result":
                            "no_change",

                        "event_id":
                            target_id,
                    }
                )

                continue

            existing_ids = {
                event.get(
                    "event_id"
                )
                for event
                in result[
                    "events"
                ]
                if event.get(
                    "event_id"
                )
            }

            replacement_id = _unique_event_id(
                replacement,
                existing_ids,
            )

            target[
                "status"
            ] = "superseded"

            target[
                "superseded_as_of"
            ] = timestamp

            target[
                "superseded_market_stage"
            ] = market_stage

            target[
                "superseded_by_event_id"
            ] = replacement_id

            target[
                "last_updated_as_of"
            ] = timestamp

            replacement.update(
                {
                    "event_id":
                        replacement_id,

                    "status":
                        "active",

                    "first_seen_as_of":
                        timestamp,

                    "last_seen_as_of":
                        timestamp,

                    "first_seen_market_stage":
                        market_stage,

                    "supersedes_event_id":
                        target_id,
                }
            )

            result[
                "events"
            ].append(
                replacement
            )

            operation_results.append(
                {
                    "action":
                        "update",

                    "result":
                        "updated",

                    "event_id":
                        target_id,

                    "replacement_event_id":
                        replacement_id,
                }
            )

            changed = True

        else:

            raise ValueError(
                "Unsupported ledger operation: "
                f"{action}"
            )

    result[
        "last_updated_as_of"
    ] = timestamp

    return (
        result,
        operation_results,
        changed,
    )
