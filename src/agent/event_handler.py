from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


TIMEZONE = "Europe/Stockholm"


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def parse_local_timestamp(
    value,
    day,
):
    """
    Accept:
        full ISO timestamp
        HH:MM
    """

    text = str(value).strip()

    if len(text) <= 8 and ":" in text:
        ts = pd.Timestamp(
            f"{day} {text}"
        )
    else:
        ts = pd.Timestamp(
            text
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


def _logistics_items(raw):
    """
    Accepted shapes:

    [
      {...},
      {...}
    ]

    {"events": [...]}

    {
      "loading_operation": {...},
      "sorting_operation": {...}
    }
    """

    if isinstance(
        raw,
        list,
    ):
        return raw

    if isinstance(
        raw,
        dict,
    ):

        if isinstance(
            raw.get("events"),
            list,
        ):
            return raw["events"]

        items = []

        for name, value in raw.items():

            if (
                str(name).startswith("_")
                or
                not isinstance(
                    value,
                    dict,
                )
            ):
                continue

            item = value.copy()

            item.setdefault(
                "name",
                name,
            )

            items.append(
                item
            )

        return items

    return []


def load_logistics_library(path):
    path = Path(
        path
    )

    if not path.exists():
        return []

    return _logistics_items(
        load_json(
            path
        )
    )


def _normalized_text(
    value,
):
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )


def find_logistics_template(
    library,
    name,
):
    """
    Resolve canonical name OR an alias.

    The LLM is asked to return the canonical name, but alias
    support keeps deterministic resolution robust.
    """

    wanted = _normalized_text(
        name
    )

    for item in library:

        candidate_names = [
            item.get(
                "name"
            ),
            item.get(
                "event_name"
            ),
            item.get(
                "id"
            ),
            item.get(
                "type"
            ),
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

        candidate_names.extend(
            aliases
        )

        normalized_candidates = {
            _normalized_text(
                candidate
            )
            for candidate
            in candidate_names
            if candidate is not None
        }

        if wanted in normalized_candidates:
            return item

    raise ValueError(
        f"Unknown logistics event/template: {name}"
    )


def logistics_catalog_for_llm(
    library,
):
    """
    Expose only OPERATIONAL semantics to the LLM.

    Important:
        unit_power_kw / power_kw are intentionally NOT exposed.

    The LLM identifies:
        process
        time window
        units OR intensity

    Deterministic code calculates electrical power later.
    """

    catalog = []

    for item in library:

        name = item.get(
            "name",
            item.get(
                "event_name"
            ),
        )

        if not name:
            continue

        default_units = item.get(
            "default_units"
        )

        unit_name = item.get(
            "unit_name"
        )

        if (
            isinstance(
                default_units,
                dict,
            )
            and
            unit_name
        ):
            scale_mode = (
                "units_or_intensity"
            )

            intensity_options = list(
                default_units.keys()
            )

        else:
            scale_mode = (
                "fixed_template"
            )

            intensity_options = []

        catalog.append(
            {
                "name":
                    str(
                        name
                    ),

                "description":
                    item.get(
                        "description"
                    ),

                "aliases":
                    item.get(
                        "aliases",
                        [],
                    ),

                "scale_mode":
                    scale_mode,

                "unit_name":
                    unit_name,

                "intensity_options":
                    intensity_options,
            }
        )

    return catalog


def _template_power_from_operational_scale(
    template,
    raw,
):
    """
    Convert operational information to electrical power.

    Priority:
        explicit operational units
        -> intensity mapped to default units
        -> fixed template power

    A directly supplied power_kw remains supported as an
    engineering override, but is not required from dispatchers.
    """

    explicit_power = raw.get(
        "power_kw"
    )

    if explicit_power is not None:

        power = float(
            explicit_power
        )

        if power <= 0:
            raise ValueError(
                "Logistics power_kw must be positive."
            )

        return {
            "power_kw":
                power,

            "units":
                raw.get(
                    "units"
                ),

            "intensity":
                raw.get(
                    "intensity"
                ),

            "unit_name":
                template.get(
                    "unit_name"
                )
                if template
                else None,

            "unit_power_kw":
                template.get(
                    "unit_power_kw"
                )
                if template
                else None,
        }

    if template is None:

        raise ValueError(
            "Logistics event requires a known "
            "logistic_event_name when power_kw is not explicitly "
            "provided."
        )

    unit_power = template.get(
        "unit_power_kw"
    )

    default_units = template.get(
        "default_units"
    )

    unit_name = template.get(
        "unit_name"
    )

    # --------------------------------------------------------
    # Unit-based operational template
    # --------------------------------------------------------

    if (
        unit_power is not None
        and
        isinstance(
            default_units,
            dict,
        )
    ):

        units = raw.get(
            "units"
        )

        intensity = raw.get(
            "intensity"
        )

        if intensity is not None:

            intensity = str(
                intensity
            ).strip().lower()

        if units is None:

            if intensity is None:

                raise ValueError(
                    f"Logistics process '{template.get('name')}' "
                    f"requires either the number of "
                    f"{unit_name or 'operational units'} "
                    "or an intensity level."
                )

            if intensity not in default_units:

                raise ValueError(
                    f"Unknown intensity '{intensity}' for "
                    f"{template.get('name')}. "
                    f"Allowed: {list(default_units.keys())}"
                )

            units = default_units[
                intensity
            ]

        units = int(
            units
        )

        if units <= 0:

            raise ValueError(
                "Logistics units must be positive."
            )

        power = (
            units
            *
            float(
                unit_power
            )
        )

        return {
            "power_kw":
                float(
                    power
                ),

            "units":
                units,

            "intensity":
                intensity,

            "unit_name":
                unit_name,

            "unit_power_kw":
                float(
                    unit_power
                ),
        }

    # --------------------------------------------------------
    # Legacy fixed-power template
    # --------------------------------------------------------

    fixed_power = template.get(
        "power_kw",
        template.get(
            "peak_kw",
            template.get(
                "load_kw"
            ),
        ),
    )

    if fixed_power is None:

        raise ValueError(
            f"Logistics template '{template.get('name')}' has no "
            "deterministic power model."
        )

    power = float(
        fixed_power
    )

    if power <= 0:

        raise ValueError(
            "Logistics template power must be positive."
        )

    return {
        "power_kw":
            power,

        "units":
            None,

        "intensity":
            raw.get(
                "intensity"
            ),

        "unit_name":
            None,

        "unit_power_kw":
            None,
    }


def get_scheduled_logistics_for_day(
    library,
    day,
):
    """
    Backward-compatible support for concrete dated entries.

    Generic operational templates are NOT auto-activated.
    """

    scheduled = []

    day_date = pd.Timestamp(
        day
    ).date()

    for item in library:

        start_value = item.get(
            "start_time",
            item.get(
                "start"
            ),
        )

        end_value = item.get(
            "end_time",
            item.get(
                "end"
            ),
        )

        if (
            start_value is None
            or
            end_value is None
        ):
            continue

        # A generic clock-time template must not auto-activate.
        if len(
            str(
                start_value
            )
        ) <= 8:
            continue

        try:

            start = parse_local_timestamp(
                start_value,
                day,
            )

            end = parse_local_timestamp(
                end_value,
                day,
            )

        except Exception:
            continue

        if start.date() != day_date:
            continue

        try:

            resolved = (
                _template_power_from_operational_scale(
                    template=item,
                    raw=item,
                )
            )

        except Exception:
            continue

        scheduled.append(
            {
                "event_type":
                    "logistics_load",

                "logistic_event_name":
                    item.get(
                        "name"
                    ),

                "start_time":
                    start.isoformat(),

                "end_time":
                    end.isoformat(),

                "power_kw":
                    resolved[
                        "power_kw"
                    ],

                "units":
                    resolved.get(
                        "units"
                    ),

                "intensity":
                    resolved.get(
                        "intensity"
                    ),

                "unit_name":
                    resolved.get(
                        "unit_name"
                    ),

                "unit_power_kw":
                    resolved.get(
                        "unit_power_kw"
                    ),

                "priority":
                    "critical",

                "flexible":
                    False,

                "source":
                    "logistic_events.json",
            }
        )

    return scheduled


def normalize_events(
    parsed_payload,
    day,
    as_of,
    event_policy,
    logistics_library_path,
):
    """
    Deterministic enrichment and validation.

    The LLM provides operational semantics.
    Physical impact is calculated here.
    """

    raw_events = parsed_payload.get(
        "events",
        [],
    )

    if not isinstance(
        raw_events,
        list,
    ):

        raise ValueError(
            "'events' must be a list."
        )

    library = load_logistics_library(
        logistics_library_path
    )

    normalized = []

    normalized.extend(
        get_scheduled_logistics_for_day(
            library,
            day,
        )
    )

    ev_policy = event_policy[
        "company_ev"
    ]

    for raw in raw_events:

        event_type = raw.get(
            "event_type"
        )

        # ====================================================
        # LOGISTICS
        # ====================================================

        if event_type == "logistics_load":

            template = None

            logistic_event_name = raw.get(
                "logistic_event_name"
            )

            if logistic_event_name:

                template = find_logistics_template(
                    library,
                    logistic_event_name,
                )

                canonical_name = template.get(
                    "name",
                    logistic_event_name,
                )

            else:

                canonical_name = None

            start_value = raw.get(
                "start_time"
            )

            end_value = raw.get(
                "end_time"
            )

            if template:

                start_value = (
                    start_value
                    or
                    template.get(
                        "start_time",
                        template.get(
                            "start"
                        ),
                    )
                )

                end_value = (
                    end_value
                    or
                    template.get(
                        "end_time",
                        template.get(
                            "end"
                        ),
                    )
                )

            if (
                start_value is None
                or
                end_value is None
            ):

                raise ValueError(
                    "logistics_load requires "
                    "start_time and end_time."
                )

            start = parse_local_timestamp(
                start_value,
                day,
            )

            end = parse_local_timestamp(
                end_value,
                day,
            )

            if end <= start:

                raise ValueError(
                    "Logistics end_time must be after start_time."
                )

            resolved = (
                _template_power_from_operational_scale(
                    template=template,
                    raw=raw,
                )
            )

            normalized_event = {
                "event_type":
                    "logistics_load",

                "start_time":
                    start.isoformat(),

                "end_time":
                    end.isoformat(),

                "power_kw":
                    resolved[
                        "power_kw"
                    ],

                "priority":
                    "critical",

                "flexible":
                    False,

                "source":
                    "llm_event",
            }

            if canonical_name:

                normalized_event[
                    "logistic_event_name"
                ] = canonical_name

            for field in [
                "units",
                "intensity",
                "unit_name",
                "unit_power_kw",
            ]:

                value = resolved.get(
                    field
                )

                if value is not None:

                    normalized_event[
                        field
                    ] = value

            normalized.append(
                normalized_event
            )

        # ====================================================
        # COMPANY EV
        # ====================================================

        elif event_type == "company_ev_charging":

            vehicles = int(
                raw[
                    "vehicles"
                ]
            )

            if vehicles <= 0:

                raise ValueError(
                    "vehicles must be positive."
                )

            arrival = parse_local_timestamp(
                raw[
                    "arrival_time"
                ],
                day,
            )

            departure = parse_local_timestamp(
                raw[
                    "departure_time"
                ],
                day,
            )

            if departure <= arrival:

                raise ValueError(
                    "EV departure_time must be after arrival_time."
                )

            energy_per_vehicle = float(
                ev_policy[
                    "energy_per_vehicle_kwh"
                ]
            )

            max_power_per_vehicle = float(
                ev_policy[
                    "max_power_per_vehicle_kw"
                ]
            )

            normalized.append(
                {
                    "event_type":
                        "company_ev_charging",

                    "vehicles":
                        vehicles,

                    "arrival_time":
                        arrival.isoformat(),

                    "departure_time":
                        departure.isoformat(),

                    "energy_per_vehicle_kwh":
                        energy_per_vehicle,

                    "max_power_per_vehicle_kw":
                        max_power_per_vehicle,

                    "total_energy_required_kwh":
                        (
                            vehicles
                            *
                            energy_per_vehicle
                        ),

                    "total_max_power_kw":
                        (
                            vehicles
                            *
                            max_power_per_vehicle
                        ),

                    "priority":
                        "flexible",

                    "flexible":
                        True,

                    "source":
                        "llm_event",
                }
            )

        else:

            raise ValueError(
                f"Unsupported event_type: {event_type}"
            )

    return {
        "day":
            day,

        "as_of":
            str(
                as_of
            ),

        "events":
            normalized,
    }
