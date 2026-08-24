# Operational AI Agent for Energy Community Scheduling

Prototype for **forecast-based day-ahead scheduling** and **natural-language operational rescheduling** of an industrial energy community with PV, battery storage, grid constraints, flexible EV charging and logistics loads.

The core design principle is:

> **The LLM interprets operational language. Deterministic Python manages state and engineering parameters. Mathematical optimization makes the scheduling decisions.**

---

## Motivation

A day-ahead energy schedule is created using information available before delivery. In real operation, new information arrives later:

```text
"charge 2 cars from 14:00 to 16:00"

"loading operation from 16:00 to 19:00"

"modify loading event"

"cancel EV charging"
```

The prototype converts these instructions into an auditable operational state change and then recalculates the energy schedule.

It distinguishes two market situations:

```text
Before DA gate closure
→ update the day-ahead schedule

After DA gate closure
→ keep the DA market commitment
→ correct the physical plan through intraday rescheduling
```

---

# Architecture

```text
Historical data
      │
      ▼
Feature engineering
      │
      ├────────────┬────────────┐
      ▼            ▼            ▼
Load forecast   PV forecast   DAM forecast
      │            │            │
      └────────────┴────────────┘
                   │
                   ▼
            DA optimization
                   │
                   ▼
             DA schedule
                   │
                   │
Stakeholder message
        │
        ▼
       LLM
        │
        ▼
Semantic event
        │
        ▼
Deterministic event/state layer
        │
        ▼
Persistent event ledger
        │
        ▼
     DA gate check
      /        \
     /          \
PRE-GATE      POST-GATE
   │              │
   ▼              ▼
DA event        ID
optimizer     optimizer
   │              │
   └──────┬───────┘
          ▼
 Current schedule
          │
          ▼
     Visualization
```

---

# Quick start

The project is developed with Python 3.12.

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
```

After the final dependency file is prepared:

```bash
pip install -r requirements.txt
```

The local LLM is accessed through LM Studio.

Example `config.env`:

```env
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
# LM_STUDIO_MODEL=<loaded-model-id>
```

---

# Available test days and intraday data scope

The complete DA + ID demonstration is currently prepared for two delivery days:

```text
2026-07-31
2026-08-01
```

These are the two days for which intraday VWAP input is included in the prototype dataset.

The intraday price data used in this repository comes from **HUPX**, not Nord Pool SE3.  
The intended Swedish SE3 intraday dataset was not available during prototype development, so HUPX VWAP data is used as a practical market-price proxy for demonstrating the ID rescheduling logic.

This is therefore a **prototype data assumption**, not a claim that HUPX prices represent actual SE3 intraday settlement.

For the intraday model, an additional simplifying assumption is used:

> From the start of intraday trading, the available VWAP is assumed to have settled at the observed level and is treated as the applicable intraday price signal for the corresponding period.

In other words, the prototype does not reconstruct the full sequence of individual intraday trades or order-book evolution. It uses VWAP as a simplified representative price for the ID rescheduling decision.

For a fully reproducible end-to-end demo, use one of the two supported delivery days above.

---

# Main commands

There are two main pipeline entry points.

## 1. Build the base day-ahead schedule

```bash
python -m src.pipelines.run_da \
  --day 2026-08-01 \
  --hpo false
```

Main file:

```text
src/pipelines/run_da.py
```

Pipeline:

```text
feature enrichment
→ load forecast
→ PV forecast
→ DAM forecast
→ base DA optimization
→ visualization
```

---

## 2. Process an operational event

```bash
python -m src.pipelines.run_event \
  --day 2026-08-01 \
  --as-of 2026-07-31T17:00:00+02:00 \
  --message "loading operation from 16:00 to 19:00"
```

Main file:

```text
src/pipelines/run_event.py
```

Pipeline:

```text
natural-language parsing
→ event handling
→ persistent ledger update
→ DA / ID routing
→ rescheduling
→ visualization
```

`--as-of` is explicit so historical simulations are reproducible and do not depend on the real system clock.

Operational timezone:

```text
Europe/Stockholm
```

Optimization resolution:

```text
15 minutes
```

---

# Files worth looking at first

If reviewing the project quickly, these are the most important files.

## `src/pipelines/run_da.py`

Orchestrates the base day-ahead workflow:

```text
forecasting
→ DA optimization
→ plot
```

---

## `src/pipelines/run_event.py`

The central operational-agent pipeline.

It handles:

```text
simulation time
DA gate closure
LLM conversation
human event selection
ledger mutation
DA / ID routing
optimizer execution
result reporting
```

This is the main file for understanding the event-driven workflow.

---

## `src/agent/llm_event_parser.py`

Natural-language interpretation.

The LLM identifies:

```text
ADD
UPDATE
CANCEL

company_ev_charging
logistics_load

time windows
vehicle count
logistics process
requested semantic changes
```

The LLM does **not** define grid limits, BESS parameters, EV energy, charger power or logistics power.

---

## `src/agent/event_handler.py`

Deterministic normalization and policy enrichment.

Examples:

```text
3 EVs
→ vehicle count from LLM
→ energy and charger power from event_policy.json

loading_operation
→ process name from LLM
→ fixed electrical load from logistic_events.json
```

---

## `src/agent/event_store.py`

Persistent operational event ledger.

Event states include:

```text
active
cancelled
superseded
```

UPDATE creates a replacement event instead of overwriting history.

---

## `src/optimization/da_optimizer.py`

Base day-ahead BESS / grid optimization.

---

## `src/optimization/da_event_optimizer.py`

Pre-gate event-aware DA optimization.

It also stores event-specific physical profiles:

```text
event_profile_<event_id>_kw
```

These profiles are used later to preserve the physical meaning of pre-gate events.

---

## `src/optimization/id_optimizer.py`

Post-gate intraday rescheduling.

It keeps the DA grid schedule as the market commitment and optimizes the required deviation.

It also preserves already executed rows from earlier ID schedules.

---

# Configuration

Main parameter files:

```text
data/parameters/
├── bess.json
├── market.json
├── event_policy.json
├── logistic_events.json
├── hp_load_fc.json
├── hp_pv_fc.json
└── hp_dam_fc.json
```

## `bess.json`

Current prototype battery is approximately:

```text
1 MW power
3 MWh energy
5–95% SOC
95% charge efficiency
95% discharge efficiency
```

The JSON file is the source of truth.

## `market.json`

Contains:

```text
grid import/export limits
15-minute timestep
DAM price unit
DA gate-closure time
```

The current prototype gate closure is:

```text
12:00 on D-1
Europe/Stockholm
```

Therefore for delivery day `2026-08-01`:

```text
2026-07-31 11:59 → PRE-GATE
2026-07-31 12:00 → POST-GATE
```

## `event_policy.json`

Contains deterministic EV policy values.

Current prototype values include:

```text
150 kWh requested energy / vehicle
100 kW maximum charging power / vehicle
```

Therefore:

```text
2 vehicles → 300 kWh request
3 vehicles → 450 kWh request
```

## `logistic_events.json`

Contains the fixed-power logistics catalog.

Current process types:

```text
loading_operation
sorting_operation
forklift_charging
cold_storage_event
```

The dispatcher only specifies the process and time window. Electrical power is resolved from the configuration file.

---

# Operational event model

Two event types are currently supported.

## Company EV charging

Flexible aggregated energy request.

Example:

```text
"charge 3 cars from 14:00 to 16:00"
```

The LLM extracts:

```text
vehicles = 3
arrival = 14:00
departure = 16:00
```

Python adds the deterministic physical policy.

The optimizer can move charging inside the given charging timerange.

Output reports:

```text
requested energy
delivered energy
unserved energy
```

Hard BESS and grid constraints are never relaxed to satisfy EV charging.

---

## Logistics load

Critical, inflexible load.

Example:

```text
"loading operation from 16:00 to 19:00"
```

The LLM extracts:

```text
loading_operation
16:00–19:00
```

Python resolves the fixed electrical power from:

```text
data/parameters/logistic_events.json
```

---

# ADD, UPDATE and CANCEL

## ADD

```bash
python -m src.pipelines.run_event \
  --day 2026-08-01 \
  --as-of 2026-07-31T10:00:00+02:00 \
  --message "charge 2 cars from 8:00 to 12:00"
```

A new event is added to the ledger and the relevant optimizer is run.

### Duplicate protection

If an identical active event is sent again:

```text
"loading operation from 16:00 to 19:00"
```

the intent remains:

```text
ADD
```

but the deterministic ledger returns:

```text
duplicate_ignored
```

No re-optimization is required because state did not change.

---

## UPDATE

```bash
python -m src.pipelines.run_event \
  --day 2026-08-01 \
  --as-of 2026-07-31T17:00:00+02:00 \
  --message "modify loading event"
```

The LLM identifies UPDATE, but it does not choose the event ID.

Python lists active events:

```text
1  ...  logistics_load
2  ...  company_ev_charging
```

The stakeholder selects the exact target.

If needed:

```text
Agent > What should be changed in this event?
```

Example:

```text
move it to 17:00 to 20:00
```

Then:

```text
old event → superseded
new event → active
```

---

## CANCEL

```bash
python -m src.pipelines.run_event \
  --day 2026-08-01 \
  --as-of 2026-07-31T17:00:00+02:00 \
  --message "cancel EV charging event"
```

The same human-in-the-loop selection is used.

The reason is intentional:

> The LLM may interpret the action, but it should not guess which persistent state object is being modified or cancelled.

---

# Market-aware routing

The event pipeline automatically decides which optimizer to use.

```text
as_of < DA gate
→ PRE-GATE
→ DA event optimizer

as_of >= DA gate
→ POST-GATE
→ ID optimizer
```

## Pre-gate

```text
new event state
→ new current DA reschedule
```

Output:

```text
data/output_data/schedules/rescheduled/da/
da_rescheduled_<DAY>.xlsx
```

## Post-gate

```text
DA grid profile
→ remains market commitment

current physical event state
→ may change

ID optimizer
→ trades / reschedules the difference
```

Intraday price input:

```text
data/input_data/id_price/id_price.xlsx
```

with:

```text
timestamp
vwap
```

The included ID VWAP data covers the two supported prototype delivery days:

```text
2026-07-31
2026-08-01
```

The VWAP values are sourced from **HUPX** and are used as a proxy intraday price signal because the corresponding Nord Pool SE3 intraday data was not available for this prototype.

The ID optimization uses the simplifying assumption that, once intraday trading is considered to have started, the relevant VWAP has settled at the available observed level. The model therefore does not simulate individual trades, bid/ask spreads or continuous order-book evolution.

Output:

```text
data/output_data/schedules/rescheduled/id/
id_rescheduled_<DAY>.xlsx
```

---

# DA commitment vs current physical plan

This is one of the main modeling ideas.

Suppose an EV event was scheduled before gate closure and is cancelled afterwards.

The correct behavior is:

```text
event
→ removed from future physical demand

DA market commitment
→ unchanged

ID optimizer
→ corrects the market position
```

The system therefore does not rewrite historical market commitments when operational information changes.

---

# Executed-history preservation

During the delivery day, multiple ID runs may occur.

Already executed periods are preserved.

Example:

```text
08:00–09:00 already happened
10:00 new event arrives
```

Then:

```text
08:00–10:00
→ preserved from current ID history

10:00 onward
→ re-optimized
```

Completed events are also removed from the normal UPDATE/CANCEL selection set.

---

# Persistent state

Operational history:

```text
data/scenarios/events/event_ledger_<DAY>.json
```

Current optimizer event snapshot:

```text
data/scenarios/events/current_optimizer_events_<DAY>.json
```

The ledger is the audit trail.

Schedules remain simple: one current DA and one current ID reschedule per delivery day.

---

# Output locations

Forecasts:

```text
data/output_data/forecast/
```

Base DA schedule:

```text
data/output_data/schedules/da/
```

Current pre-gate schedule:

```text
data/output_data/schedules/rescheduled/da/
```

Current post-gate schedule:

```text
data/output_data/schedules/rescheduled/id/
```

Plots:

```text
data/output_data/plots/
```

Typical plot panels show:

```text
load
PV
electricity price
battery charge/discharge
SOC
resulting grid consumption
```

---

# Suggested demo

A short demo can show almost the complete architecture.

## 1. Base DA schedule

```bash
python -m src.pipelines.run_da \
  --day 2026-08-01 \
  --hpo false
```

## 2. Pre-gate logistics ADD

```bash
python -m src.pipelines.run_event \
  --day 2026-08-01 \
  --as-of 2026-07-31T10:30:00+02:00 \
  --message "loading operation from 16:00 to 19:00"
```

Shows:

```text
LLM interpretation
→ deterministic logistics mapping
→ PRE-GATE routing
→ DA rescheduling
```

## 3. Post-gate EV ADD

```bash
python -m src.pipelines.run_event \
  --day 2026-08-01 \
  --as-of 2026-07-31T17:10:00+02:00 \
  --message "ev charging event from 14:00 to 16:00, there are 2 cars which should be charged"
```

Shows:

```text
POST-GATE routing
→ 300 kWh EV request
→ ID rescheduling
```

## 4. UPDATE

```bash
python -m src.pipelines.run_event \
  --day 2026-08-01 \
  --as-of 2026-07-31T17:20:00+02:00 \
  --message "ev charging event should be modified, there will be 3 cars instead of 2"
```

Select the EV event.

Expected:

```text
2 vehicles → 3 vehicles
300 kWh → 450 kWh
old event → superseded
replacement → active
```

This one sequence demonstrates the LLM, persistent state, human-in-the-loop handling and intraday optimizer.

---

# Clean demo reset

Before a completely fresh demonstration:

```bash
rm -f data/scenarios/events/event_ledger_{date}.json
rm -f data/scenarios/events/current_optimizer_events_{date}.json

rm -f data/output_data/schedules/rescheduled/da/da_rescheduled_{date}.xlsx
rm -f data/output_data/schedules/rescheduled/id/id_rescheduled_{date}.xlsx
```

Then rebuild the base DA schedule.

---

# Scope and assumptions

This is a research/interview prototype, not a production EMS.

Current simplifications include:

```text
15-minute resolution
point forecasts
aggregated EV charging
fixed-power logistics templates
simplified symmetric ID price treatment
HUPX VWAP used as an ID price proxy instead of Nord Pool SE3
VWAP assumed to be settled at the observed level from ID trading start
ID demo data limited to 2026-07-31 and 2026-08-01
no live SCADA connection
no live market API
one current schedule per stage/day
```

The goal is to demonstrate the architecture and decision flow clearly rather than reproduce every detail of a commercial energy-management system.

---

# Summary

The complete prototype flow is:

```text
forecast
→ base DA schedule
→ natural-language operational event
→ semantic interpretation
→ deterministic state update
→ DA / ID market routing
→ constrained optimization
→ current schedule
→ visualization
→ persistent audit history
```

The key architectural idea is:

> **Use the LLM for language, deterministic software for state and engineering assumptions, and optimization for physical and market decisions.**
