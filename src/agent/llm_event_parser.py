from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(
    PROJECT_ROOT / "config.env"
)

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
TIMEZONE = "Europe/Stockholm"


def get_base_url():
    return os.getenv(
        "LM_STUDIO_BASE_URL",
        DEFAULT_BASE_URL,
    ).rstrip("/")


def get_model():
    configured_model = os.getenv(
        "LM_STUDIO_MODEL"
    )

    if configured_model:
        return configured_model

    response = requests.get(
        f"{get_base_url()}/models",
        timeout=15,
    )

    response.raise_for_status()

    models = response.json().get(
        "data",
        [],
    )

    if not models:

        raise RuntimeError(
            "LM Studio is reachable, but no model is loaded."
        )

    return models[0][
        "id"
    ]


def clean_json_text(
    text,
):
    text = text.strip()

    if text.startswith(
        "```"
    ):

        text = (
            text
            .strip(
                "`"
            )
            .strip()
        )

        if text.lower().startswith(
            "json"
        ):

            text = text[
                4:
            ].strip()

    return text


def _compact_known_events(
    known_events,
):
    """
    Send semantic event information to the LLM.

    Physical coefficients such as unit_power_kw are intentionally
    omitted.
    """

    compact = []

    for event in known_events or []:

        item = {
            "event_id":
                event.get(
                    "event_id"
                ),

            "event_type":
                event.get(
                    "event_type"
                ),
        }

        if (
            event.get(
                "event_type"
            )
            ==
            "company_ev_charging"
        ):

            item.update(
                {
                    "vehicles":
                        event.get(
                            "vehicles"
                        ),

                    "arrival_time":
                        event.get(
                            "arrival_time"
                        ),

                    "departure_time":
                        event.get(
                            "departure_time"
                        ),
                }
            )

        elif (
            event.get(
                "event_type"
            )
            ==
            "logistics_load"
        ):

            item.update(
                {
                    "logistic_event_name":
                        event.get(
                            "logistic_event_name"
                        ),

                    "start_time":
                        event.get(
                            "start_time"
                        ),

                    "end_time":
                        event.get(
                            "end_time"
                        ),

                }
            )

        compact.append(
            item
        )

    return compact


def _resolve_llm_timestamp(
    value,
    day,
):
    """
    Deterministically convert an LLM time value into a real
    Europe/Stockholm ISO timestamp.
    """

    if value is None:
        return value

    value = str(
        value
    ).strip()

    if (
        "DELIVERY_DAY"
        in value
        or
        "STOCKHOLM_OFFSET"
        in value
    ):

        match = re.search(
            r"T(\d{1,2}:\d{2}(?::\d{2})?)",
            value,
        )

        if not match:

            raise ValueError(
                "LLM returned an unresolved timestamp "
                f"placeholder: {value}"
            )

        value = match.group(
            1
        )

    if re.fullmatch(
        r"\d{1,2}:\d{2}(?::\d{2})?",
        value,
    ):

        return (
            pd.Timestamp(
                f"{day} {value}"
            )
            .tz_localize(
                TIMEZONE
            )
            .isoformat()
        )

    try:

        ts = pd.Timestamp(
            value
        )

    except Exception as exc:

        raise ValueError(
            "LLM returned an invalid timestamp: "
            f"{value}"
        ) from exc

    if ts.tzinfo is None:

        ts = ts.tz_localize(
            TIMEZONE
        )

    else:

        ts = ts.tz_convert(
            TIMEZONE
        )

    return ts.isoformat()


def _normalize_operation_timestamps(
    operations,
    day,
):

    for operation in operations:

        action = str(
            operation.get(
                "action",
                "",
            )
        ).lower()

        if action == "add":

            event = operation.get(
                "event"
            )

        elif action == "update":

            event = operation.get(
                "changes"
            )

        else:

            event = None

        if not isinstance(
            event,
            dict,
        ):

            continue

        # ADD events can be normalized from event_type.
        # UPDATE changes intentionally contain only a patch, so
        # normalize any known time field by its field name.
        fields = [
            field
            for field
            in [
                "arrival_time",
                "departure_time",
                "start_time",
                "end_time",
            ]
            if field in event
        ]

        for field in fields:

            event[
                field
            ] = _resolve_llm_timestamp(
                event[
                    field
                ],
                day,
            )

    return operations


def _canonicalize_logistics_names(
    operations,
    logistics_catalog,
):
    """
    Deterministically map canonical names and aliases.

    Unknown process names cause a clarification request rather
    than allowing the LLM to invent a template.
    """

    alias_map = {}

    canonical_names = []

    for item in logistics_catalog or []:

        canonical = str(
            item[
                "name"
            ]
        )

        canonical_names.append(
            canonical
        )

        candidates = [
            canonical
        ]

        aliases = item.get(
            "aliases",
            [],
        )

        if isinstance(
            aliases,
            str,
        ):

            aliases = [
                aliases
            ]

        candidates.extend(
            aliases
        )

        for candidate in candidates:

            key = (
                str(
                    candidate
                )
                .strip()
                .lower()
                .replace(
                    "-",
                    " ",
                )
                .replace(
                    "_",
                    " ",
                )
            )

            alias_map[
                key
            ] = canonical

    for operation in operations:

        action = operation.get(
            "action"
        )

        if action == "add":

            event = operation.get(
                "event"
            )

        elif action == "update":

            event = operation.get(
                "changes"
            )

        else:

            event = None

        if not isinstance(
            event,
            dict,
        ):

            continue

        # For ADD, only logistics_load needs catalog resolution.
        # For UPDATE, resolve only when logistic_event_name itself
        # is being changed.
        if action == "add":

            if (
                event.get(
                    "event_type"
                )
                !=
                "logistics_load"
            ):

                continue

        elif action == "update":

            if (
                "logistic_event_name"
                not in event
            ):

                continue

        process_name = event.get(
            "logistic_event_name"
        )

        # An explicit measured power override remains backward
        # compatible, but dispatchers normally use named processes.
        if (
            not process_name
            and
            event.get(
                "power_kw"
            )
            is not None
        ):

            continue

        if not process_name:

            return (
                False,
                (
                    "Which logistics process is this? "
                    "Please name the operational process."
                ),
            )

        key = (
            str(
                process_name
            )
            .strip()
            .lower()
            .replace(
                "-",
                " ",
            )
            .replace(
                "_",
                " ",
            )
        )

        canonical = alias_map.get(
            key
        )

        if canonical is None:

            available = ", ".join(
                canonical_names
            )

            return (
                False,
                (
                    "I could not match that logistics process "
                    "to the operational catalog. Available "
                    f"processes: {available}."
                ),
            )

        event[
            "logistic_event_name"
        ] = canonical

    return (
        True,
        None,
    )



def _recover_pending_update_intent(
    parsed,
    conversation,
):
    """
    Small local models sometimes return needs_clarification for:

        "modify loading event"
        "update the EV event"
        "change the logistics event"

    In the human-in-the-loop architecture this should NOT trigger
    a clarification yet. Event selection comes first.

    This recovery is safe because it performs no state mutation:
    it only opens the deterministic event-selection flow.
    """

    if (
        parsed.get(
            "status"
        )
        !=
        "needs_clarification"
    ):

        return parsed

    latest = ""

    for message in reversed(
        conversation
    ):

        if (
            message.get(
                "role"
            )
            ==
            "user"
        ):

            latest = str(
                message.get(
                    "content",
                    "",
                )
            ).lower()

            break

    if not latest:

        return parsed

    update_terms = [
        "modify",
        "modified",
        "update",
        "change",
        "reschedule",
        "move",
        "módos",
        "modos",
        "változt",
        "valtozt",
        "átrak",
        "atrak",
        "áttenn",
        "attenn",
    ]

    event_terms = [
        "event",
        "loading",
        "logistics",
        "sorting",
        "forklift",
        "cold storage",
        "ev",
        "charging",
        "charge",
        "tölt",
        "tolt",
        "rakod",
    ]

    if (
        any(
            term in latest
            for term
            in update_terms
        )
        and
        any(
            term in latest
            for term
            in event_terms
        )
    ):

        return {
            "status":
                "actionable",

            "question":
                None,

            "operations":
                [
                    {
                        "action":
                            "update",

                        "changes":
                            {},
                    }
                ],
        }

    return parsed



def _has_explicit_update_language(
    text,
):
    message = str(
        text
    ).strip().lower()

    terms = [
        "modify",
        "modified",
        "update",
        "updated",
        "change",
        "changed",
        "move",
        "moved",
        "reschedule",
        "rescheduled",
        "instead of",
        "make it",
        "actually",
        "new period",
        "new time",
        "revised",
        "correction",
        "correct",
        "módos",
        "modos",
        "változt",
        "valtozt",
        "átrak",
        "atrak",
        "áttenn",
        "attenn",
    ]

    return any(
        term in message
        for term
        in terms
    )


def _has_explicit_cancel_language(
    text,
):
    message = str(
        text
    ).strip().lower()

    terms = [
        "cancel",
        "cancelled",
        "canceled",
        "there will be no",
        "will not happen",
        "won't happen",
        "no longer",
        "remove",
        "delete",
        "elmarad",
        "nem lesz",
        "töröl",
        "torol",
    ]

    return any(
        term in message
        for term
        in terms
    )


def _recover_neutral_event_as_add(
    parsed,
    conversation,
):
    """
    Prevent the LLM from turning a plain event declaration into
    UPDATE merely because a similar event already exists.

    This is intentionally conservative:
    - explicit UPDATE language -> keep UPDATE
    - explicit CANCEL language -> keep CANCEL
    - otherwise, if the model returned UPDATE with a complete
      event-like semantic payload, reinterpret it as ADD only when
      the latest user message is a neutral declaration.

    Duplicate suppression remains the ledger's responsibility.
    """

    latest = ""

    for message in reversed(
        conversation
    ):

        if (
            message.get(
                "role"
            )
            ==
            "user"
        ):

            latest = str(
                message.get(
                    "content",
                    "",
                )
            )

            break

    if not latest:
        return parsed

    if (
        _has_explicit_update_language(
            latest
        )
        or
        _has_explicit_cancel_language(
            latest
        )
    ):

        return parsed

    operations = parsed.get(
        "operations",
        [],
    )

    if (
        parsed.get(
            "status"
        )
        !=
        "actionable"
        or
        len(
            operations
        )
        !=
        1
    ):

        return parsed

    operation = operations[
        0
    ]

    if (
        str(
            operation.get(
                "action",
                "",
            )
        ).lower()
        !=
        "update"
    ):

        return parsed

    changes = operation.get(
        "changes",
        {},
    )

    if not isinstance(
        changes,
        dict,
    ):

        return parsed

    # A neutral logistics declaration usually contains the process
    # and a full time window. Rebuild an ADD event envelope.
    if (
        "start_time"
        in changes
        and
        "end_time"
        in changes
    ):

        event = {
            "event_type":
                "logistics_load",

            "start_time":
                changes[
                    "start_time"
                ],

            "end_time":
                changes[
                    "end_time"
                ],
        }

        if (
            "logistic_event_name"
            in changes
        ):

            event[
                "logistic_event_name"
            ] = changes[
                "logistic_event_name"
            ]

        else:

            # The LLM may have omitted the canonical name because
            # it thought this was an UPDATE. Let the parser's normal
            # logistics validation request clarification rather than
            # guessing a process.
            return parsed

        return {
            "status":
                "actionable",

            "question":
                None,

            "operations":
                [
                    {
                        "action":
                            "add",

                        "event":
                            event,
                    }
                ],
        }

    # Same idea for a complete EV declaration.
    if (
        "vehicles"
        in changes
        and
        "arrival_time"
        in changes
        and
        "departure_time"
        in changes
    ):

        return {
            "status":
                "actionable",

            "question":
                None,

            "operations":
                [
                    {
                        "action":
                            "add",

                        "event":
                            {
                                "event_type":
                                    "company_ev_charging",

                                "vehicles":
                                    changes[
                                        "vehicles"
                                    ],

                                "arrival_time":
                                    changes[
                                        "arrival_time"
                                    ],

                                "departure_time":
                                    changes[
                                        "departure_time"
                                    ],
                            },
                    }
                ],
        }

    return parsed


def parse_event_conversation(
    conversation,
    day,
    as_of,
    known_events=None,
    logistics_catalog=None,
):
    """
    Semantic operational-event parser.

    Supported actions:
        add
        cancel
        update
    """

    system_prompt = """
You are the operational event intake agent for an industrial
energy community.

The stakeholder may speak Hungarian or English.

Your job is ONLY to understand operational instructions.

You do NOT optimize.
You do NOT calculate electrical power.
You do NOT invent physical parameters.
You do NOT invent event IDs.


============================================================
OUTPUT
============================================================

Return JSON only:

{
  "status": "actionable | needs_clarification | no_action",
  "question": "string or null",
  "operations": []
}


Operation forms:

IMPORTANT:
Every object inside "operations" MUST contain the "action" field.

ADD
{
  "action": "add",
  "event": {...}
}

CANCEL
{
  "action": "cancel"
}

UPDATE
{
  "action": "update",
  "changes": {
    "... only fields explicitly changed by the stakeholder ..."
  }
}

If the stakeholder clearly asks to MODIFY / UPDATE / CHANGE an
existing event but has not yet said WHAT should change, this is
still an actionable UPDATE intent.

Return:

{
  "status": "actionable",
  "question": null,
  "operations": [
    {
      "action": "update",
      "changes": {}
    }
  ]
}

Do NOT ask which event to modify.
Do NOT ask what should be changed at this stage.

The deterministic Python layer will FIRST let the stakeholder
select the exact event and THEN ask what should be changed.


============================================================
EVENT SELECTION RESPONSIBILITY
============================================================

You receive KNOWN_ACTIVE_EVENTS for conversational context only.

CRITICAL RULE:

For CANCEL and UPDATE you MUST NOT choose, copy, guess, or return
an event_id.

The deterministic Python layer will display the active event list
to the stakeholder and let the stakeholder select the exact event.

Your responsibility is only:

- detect ADD / CANCEL / UPDATE / no_action;
- for UPDATE, extract only the requested semantic changes;
- for ADD, extract the complete new semantic event.

CRITICAL INTENT RULE:

A neutral declaration of an operational event is ALWAYS ADD,
even if a similar or identical active event already exists.

Examples:

"loading operation from 16:00 to 19:00"
→ ADD

"EV charging from 14:00 to 16:00 for 2 cars"
→ ADD

Do NOT infer UPDATE merely because KNOWN_ACTIVE_EVENTS already
contains a similar event.

Duplicate detection is NOT your responsibility.
The deterministic ledger handles exact duplicates.

Use UPDATE only when the stakeholder explicitly expresses change,
for example with language such as:

- modify
- update
- change
- move
- reschedule
- instead of
- make it
- actually
- new period
- revised
- módosít
- változtat
- átrak

Do not ask which event ID should be used.
Do not claim that an event has a particular ID.
Do not count or classify event IDs yourself.

Examples:

"cancel the loading operation"
→ {"action": "cancel"}

"there will be no EV charging"
→ {"action": "cancel"}

"move the loading operation to 17:00-19:00"
→ {
     "action": "update",
     "changes": {
       "start_time": "...17:00...",
       "end_time": "...19:00..."
     }
   }

"change the EV charging to 4 vehicles"
→ {
     "action": "update",
     "changes": {
       "vehicles": 4
     }
   }

If the stakeholder clearly requests an UPDATE but has not yet
specified the new values, return an empty "changes": {} object.

Do NOT ask for the new values yet. Python will first show the active
event list, let the stakeholder select one event, and only then ask
for the requested changes.

Event identity ambiguity is NOT your job; Python handles it.


============================================================
COMPANY EV CHARGING
============================================================

event_type:
"company_ev_charging"

Required:
- vehicles
- arrival_time
- departure_time

arrival_time and departure_time are the charging availability
window boundaries.

"charge 200 cars from 18:00 to 22:00"
is actionable without clarification.

Do NOT invent:
- energy per vehicle
- charging power
- priorities

Those come from deterministic policy.


============================================================
LOGISTICS
============================================================

The dispatcher is NOT expected to know electrical power.

The stakeholder only describes:

- the logistics process
- start time
- end time

You receive KNOWN_LOGISTICS_PROCESSES.

Each known process has a fixed deterministic electrical power
defined outside the LLM.

For a known logistics process return:

{
  "event_type": "logistics_load",
  "logistic_event_name": "<EXACT CANONICAL NAME>",
  "start_time": "...",
  "end_time": "..."
}

Example:

"loading operation from 14:00 to 16:00"

Return:

{
  "event_type": "logistics_load",
  "logistic_event_name": "loading_operation",
  "start_time": "...14:00...",
  "end_time": "...16:00..."
}

Do NOT ask:
- how many docks
- how many forklifts
- low / medium / high intensity
- electrical power in kW

Do NOT return:
- units
- intensity
- unit_power_kw

NEVER invent power_kw for a known logistics process.

The deterministic event-handler resolves the fixed power from
logistic_events.json.

If the stakeholder explicitly gives a measured power for an
unknown external process, power_kw may still be used as an
engineering override. For known catalog processes, always prefer
the named fixed-power template.


============================================================
UPDATE
============================================================

For UPDATE return ONLY the semantic fields that the stakeholder
explicitly wants to change inside "changes".

Do NOT construct a complete replacement event.
Do NOT choose the target event.
Do NOT return event_id.

Allowed EV change fields:
- vehicles
- arrival_time
- departure_time

Allowed logistics change fields:
- logistic_event_name
- start_time
- end_time
- power_kw only when the stakeholder explicitly gives a measured
  electrical power override

Examples:

"move it to 17:00-19:00"
→ changes = {
    "start_time": "...17:00...",
    "end_time": "...19:00..."
  }

"make it 4 cars instead"
→ changes = {
    "vehicles": 4
  }

The deterministic Python layer merges these changes into the
stakeholder-selected active event.


============================================================
TIME
============================================================

Use the supplied delivery day when only clock times are given.

Use real ISO timestamps. Never output literal placeholders such
as DELIVERY_DAY or STOCKHOLM_OFFSET.


============================================================
NO ACTION
============================================================

Use status = "no_action" for greetings, thanks, or messages that
do not request an operational change.
""".strip()

    known_events_payload = json.dumps(
        _compact_known_events(
            known_events
        ),
        indent=2,
        ensure_ascii=False,
    )

    logistics_payload = json.dumps(
        logistics_catalog
        or [],
        indent=2,
        ensure_ascii=False,
    )

    messages = [
        {
            "role":
                "system",

            "content":
                system_prompt,
        },
        {
            "role":
                "system",

            "content":
                (
                    f"Delivery day: {day}\n"
                    f"Simulation as-of time: {as_of}\n\n"
                    "KNOWN_ACTIVE_EVENTS:\n"
                    f"{known_events_payload}\n\n"
                    "KNOWN_LOGISTICS_PROCESSES:\n"
                    f"{logistics_payload}"
                ),
        },
    ]

    messages.extend(
        conversation
    )

    response = requests.post(
        f"{get_base_url()}/chat/completions",
        json={
            "model":
                get_model(),

            "temperature":
                0.0,

            "messages":
                messages,
        },
        timeout=120,
    )

    response.raise_for_status()

    raw_text = (
        response.json()
        ["choices"][0]
        ["message"]
        ["content"]
    )

    text = clean_json_text(
        raw_text
    )

    try:

        parsed = json.loads(
            text
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "LLM response was not valid JSON:\n"
            f"{text}"
        ) from exc

    parsed = _recover_pending_update_intent(
        parsed=parsed,
        conversation=conversation,
    )

    parsed = _recover_neutral_event_as_add(
        parsed=parsed,
        conversation=conversation,
    )

    status = parsed.get(
        "status"
    )

    if status not in {
        "actionable",
        "needs_clarification",
        "no_action",
    }:

        raise ValueError(
            "Invalid LLM status: "
            f"{status}"
        )

    operations = parsed.get(
        "operations",
        [],
    )

    if not isinstance(
        operations,
        list,
    ):

        raise ValueError(
            "'operations' must be a list."
        )

    operations = (
        _normalize_operation_timestamps(
            operations=operations,
            day=day,
        )
    )

    parsed[
        "operations"
    ] = operations

    # --------------------------------------------------------
    # Operation envelope validation
    # --------------------------------------------------------

    for operation in operations:

        action = str(
            operation.get(
                "action",
                "",
            )
        ).strip().lower()

        # Small local models occasionally omit the explicit action
        # field. Repair only unambiguous shapes.
        if not action:

            if isinstance(
                operation.get(
                    "event"
                ),
                dict,
            ):

                action = "add"

            elif isinstance(
                operation.get(
                    "changes"
                ),
                dict,
            ):

                action = "update"

        if action not in {
            "add",
            "cancel",
            "update",
        }:

            raise ValueError(
                "Unsupported LLM operation shape. "
                f"Received: {operation}"
            )

        operation[
            "action"
        ] = action

        if action == "add":

            if not isinstance(
                operation.get(
                    "event"
                ),
                dict,
            ):

                raise ValueError(
                    "ADD requires event."
                )

        elif action == "cancel":

            # Event selection belongs to deterministic Python.
            operation.pop(
                "event_id",
                None,
            )

        elif action == "update":

            operation.pop(
                "event_id",
                None,
            )

            operation.pop(
                "replacement_event",
                None,
            )

            changes = operation.get(
                "changes"
            )

            if changes is None:

                changes = {}

                operation[
                    "changes"
                ] = changes

            if not isinstance(
                changes,
                dict,
            ):

                raise ValueError(
                    "UPDATE 'changes' must be an object."
                )

            # Empty changes is intentionally valid here.
            # The human-in-the-loop layer will first select the
            # target event and then ask for the actual modification.

    # --------------------------------------------------------
    # Deterministic logistics name validation
    # --------------------------------------------------------

    if status == "actionable":

        valid, question = (
            _canonicalize_logistics_names(
                operations=operations,
                logistics_catalog=(
                    logistics_catalog
                    or []
                ),
            )
        )

        if not valid:

            return {
                "status":
                    "needs_clarification",

                "question":
                    question,

                "operations":
                    [],
            }

    if (
        status == "actionable"
        and
        not operations
    ):

        raise ValueError(
            "LLM returned actionable status "
            "without operations."
        )

    if (
        status
        ==
        "needs_clarification"
        and
        not parsed.get(
            "question"
        )
    ):

        parsed[
            "question"
        ] = (
            "Please provide the missing "
            "operational details."
        )

    if status == "no_action":

        parsed[
            "question"
        ] = None

        parsed[
            "operations"
        ] = []

    return parsed


def parse_event_message(
    message,
    day,
    as_of,
    known_events=None,
    logistics_catalog=None,
):

    return parse_event_conversation(
        conversation=[
            {
                "role":
                    "user",

                "content":
                    message,
            }
        ],
        day=day,
        as_of=as_of,
        known_events=known_events,
        logistics_catalog=logistics_catalog,
    )
