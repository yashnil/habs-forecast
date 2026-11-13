# Neritic v15: the trip lifecycle

A trip is the reader's own plan for one departure: a port, a fishery, an intended date, their own limits, their
economics, the evidence Neritic showed when they planned, later rechecks, and afterwards what happened. It lives in
this browser only (`nx-trips-v1:*`); nothing is sent to Neritic. Nothing in a trip is a verdict: no state or field
says a trip is safe, legal, approved or profitable.

Code: `coastwatch-web/src/nx/trips/` (contract `types.ts`, lifecycle `lifecycle.ts`), the planner in
`src/nx/coast/` (`plan.ts`, `PlanFields.tsx`), the page `src/app/(neritic)/coast/trips/`.

## States

```
            plan (port + fishery + date)          complete / record outcome
   Draft ─────────────────────────────▶ Planned ─────────────────────────────▶ Completed
     │  ▲                                │  ▲
     │  └──── loses its date ────────────┘  │
     │ cancel                    cancel │   │ reopen (has a date)
     ▼                                  ▼   │
   Cancelled ───────── reopen (no date) ──▶ Draft
```

| State | Means | Can change |
|---|---|---|
| Draft | Saved without everything a plan needs (usually the date) | everything |
| Planned | Port, fishery and date set; the departure is ahead or just past | everything; losing the date makes it a Draft |
| Completed | The reader marked it done; the outcome may still be recorded or corrected | outcome, name, notes (the plan is kept as it was) |
| Cancelled | Set aside; plan, snapshots and numbers kept | name, notes; reopen |

A **recheck is an action, never a state**. The planning snapshot is never overwritten (a refresh moves the old one to
`superseded`); at most 8 rechecks are kept.

## The one next action (`nextAction`)

The workspace shows one filled button, chosen in this order:

1. Cancelled: **Reopen this trip**.
2. Completed: **Record outcome** until one is recorded, then **Edit outcome** (planned vs actual leads).
3. Draft: **Plan this trip** (opens the plan to set a date, or plans it when it has one).
4. Planned, departure date passed: **Record outcome**.
5. Meaningful changes since planning (priority 1–2) not yet opened: **Review changes**.
6. Departure today or tomorrow, with no recheck in the last 12 hours or an incomplete one: **Recheck before departure**.
7. No personal limits set: **Edit assumptions**.
8. Departure inside the forecast's reach (about 5 days): **Compare departures**; further out: **Edit assumptions**.

## Screens and the question each answers

| Screen | Address | Answers |
|---|---|---|
| Planner (brief) | `/coast` | Where, what, when? Port → fishery → date (optional time, your own limits, area words) → the gates and sea window → optional expenses → **Save trip**. No date: kept as a draft. |
| Planner (worksheet) | `/coast?view=trip` | What would it pay? The v13/v14 worksheet. With a date set, the one button is **Save trip**; without, it keeps the v13 saved-worksheet behaviour. |
| Trip Workspace | `/coast/trips?id=…` | Where, what, when; what changed since I planned; what still needs checking; what next. Ticket beside the port: heading, "Since you planned this trip" (the change feed), the one filled next action, then three quiet words (Edit plan, Briefing, More: mark completed, cancel, all trips); stub: the official line as shown at planning (its verified state a glyph), net to boat with its break-even, "kept on this device". Dock: the departure timeline (select two periods to compare them). Evidence (the snapshot, recheck details, the full economics, edit plan, outcome) opens in drawers. |
| Briefing | `/coast/trips?id=…&view=briefing` | What do I take to the boat? One printable page from this device's copy, offline. |
| History | `/coast/trips` | What have I planned and done? Upcoming, drafts, completed, cancelled; empty: one line and **Plan a trip**. |

Phone: the same order in the sheet; heading, changes, the next action pinned at the foot of the ticket, the timeline,
then the rest. Completed trips lead with planned vs actual; cancelled trips are quiet.

## Rules kept

- Save trip takes the planning snapshot from what the page already loaded (no second fetch); saving with no economics
  is fine (conditions and official information only).
- Personal limits are labelled as the reader's own, never as safe limits; blank is "none set", never 0.
- Area is a few approximate words, never coordinates.
- "Since you planned" never says nothing changed when a source could not be checked; offline compares against the
  newest saved copy and says so.
- Money only through the engine of record (`src/lib/tripEvaluation.ts`); "net to boat", never "profit".
