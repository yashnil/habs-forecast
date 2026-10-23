# My Coast and trip economics: audit (v10)

Case studied: Santa Cruz (port-593), commercial, salmon, on staging data at `http://localhost:3417/coast?place=port-593&for=commercial&species=salmon`, 2026-10-22.
Code read: `src/lib/tripEconomics.ts` (sha256 `9255ca16…`), `src/v4/coast/economics/{TripEconomics,Worksheet,draft}.tsx|ts`, `src/content/fisheryPresets.ts`, `src/content/referenceValues.ts`, `src/components/coast/FishingBrief.tsx`, `src/v4/coast/Briefing.tsx`, `src/v4/ui/Fields.tsx`, and the design history in `docs/coastwatch/v6/plan.md` and `docs/coastwatch/v7/design.md`.
Evidence: a Playwright run that fills the calculator the way a user would (`scratchpad/v10-econ/trace.mjs`, log in `scratchpad/v10-econ/trace/trace.json`, screenshots in `scratchpad/v10-econ/trace/`).

## 1. Why Net, Cost and Break-even show dashes

**Root cause.** In the owner's state, catch, price and fuel price are filled in. Two inputs are still blank, and neither has a field in the visible part of the panel:

1. **Fuel quantity.** The Salmon troll preset fills only the burn *rates*: steaming 7.8 gal/h and fishing 3.6 gal/h, both from NWFSC-121 Table 31. It leaves round-trip distance, cruising speed, hours fishing and direct gallons blank. `fuelGallons()` returns `null` when neither the steaming term (nm, kt and gph) nor the fishing term (hours and gph) is complete. So `fuelUsd`, `tripCostUsd` and everything after them are `null`, and `missing` gets `"fuel"`. A user sees burn rates already filled and fairly concludes that fuel is handled. It is not.
2. **Crew share.** The salmon-troll preset has `crewPct: null`, because NWFSC-121 has no salmon-specific share, so the draft starts as `{kind: "afterExpenses", pct: null}` and `missing` gets `"crewShare"`. The crew model *looks* chosen, since the select reads "Share after expenses", but its parameter is blank. `netUsd` and all three break-evens need `crewKnown`.

Every field that would clear either item sits inside the collapsed **Full worksheet** disclosure (`econ-advanced`). The quick block shows only fishery, vessel, fuel price, catch and price. The only explanation is one grey line: "Needs: fuel use (gallons, or distance, speed and burn); crew share." It names no field, links nowhere, and does not say that the fields are folded below.

The trace, with each step done as a user would:

| Step | Net | Cost | Break-even | "Needs" line |
|---|---|---|---|---|
| 0 fresh (preset applied; burn 7.8 / 3.6 prefilled; crew % blank) | — | — | — | fuel use; pounds; price; crew share |
| 1 catch 400 lb | — | — | — | fuel use; price; crew share |
| 2 price $9/lb | — | — | — | fuel use; crew share |
| 3 tap "Use $8.242" (EIA): **the owner's state** | — | — | — | fuel use; crew share |
| 3b open "Breakdown and sensitivity" | — | — | — | Gross **$3,600** shows here, and only here |
| 4 (in Full worksheet) hours fishing 10 | — | **$297** | — | crew share |
| 5 round trip 40 nm, no speed yet | — | **— (regresses)** | — | fuel use (generic again) |
| 6 cruise 8 kt | — | $618 | — | crew share |
| 7 crew share 30 % | **+$2,087** | $618 | 69 lb | — |
| 8 switch to Day rate, no parameters | — | $618 | — | crew day rate |
| 9 2 crew × $250 × 1 day | +$2,482 | $618 | 124 lb | — |
| 10 back to after-expenses; gallons 60 directly | +$2,174 | $495 | 55 lb | — |

Three defects show up beyond the root cause:
- **Step 4 makes a silent assumption.** With hours fishing filled and the round trip blank, the module counts steaming as zero (`steamNone`) and shows $297 as the "Trip cost". Nothing says that steaming is excluded.
- **Step 5 is a regression.** Adding the distance before the speed removes a result that was already on screen, and the message falls back to the generic fuel text without naming "cruising speed".
- **Breakdown placement.** `Breakdown and sensitivity` sits above the inputs but is collapsed, so the one partial result the module already computes (gross) is hidden while the headline row shows three dashes.

## 2. Dependency table (from `computeTrip`)

"Blank = 0" marks inputs the module treats as zero rather than as missing.

| Output | Required inputs | Treated as 0 when blank | What the UI shows when missing |
|---|---|---|---|
| `fuelGal` | `fuelGal`, **or** (nm, kt > 0, steamGph) and/or (hours, fishingGph) | the other leg, when one leg is complete | Only a hint inside Full worksheet ("Worked out: N gal"); otherwise nothing |
| `fuelUsd` | fuelGal and fuelUsdPerGal (or 0 gal) | — | **Never shown on its own anywhere** |
| `tripCostUsd` | fuelUsd | every cost line, bait | Result line "Trip cost —" |
| `grossUsd` | lb and $/lb on **every** catch line | — | Breakdown row only (collapsed) |
| `totalLb` | lb on every line | — | not shown |
| `feesUsd` | gross and totalLb | fee % and $/lb | Breakdown row, only when > 0 |
| `crewUsd` | after-expenses: everything net needs; of-gross: gross and pct; day rate: crew, $/day and days only | — | Breakdown "Crew —" |
| `netUsd` | gross, totalLb, crew parameters, fuelUsd | costs, fees | "Net to boat —" |
| `breakEvenLb` | primary **price** (not its lb), crew, fuelUsd, other lines complete | costs, fees | "Break-even —" |
| `breakEvenUsdPerLb` | primary **lb** (not its price), crew, fuelUsd, other lines | costs, fees | Used as a fallback inside the same "Break-even" slot (mixes lb and $/lb) |
| `breakEvenFuelUsdPerGal` | everything net needs, plus fuelGal | — | Sentence inside the sensitivity box |
| `sensitivity` (3×3), `fuelSensitivity` | everything net needs | — | **Whole box absent**, with no placeholder |
| `missing` | — | — | One text line, four category keys (`fuel`, `fuelPrice`, `catch`, `price`, `crewShare`, `dayRate`) and no field identity |

**Computable from partial inputs, but hidden or not offered:**
- **Gross** (catch × price) appears as soon as both are known, but it is shown only in the collapsed breakdown.
- **Fuel gallons and fuel $** appear as soon as the fuel inputs are known. Fuel $ is never displayed as its own figure.
- **Break-even catch does not need the user's catch.** It needs price, fuel and crew. The UI never says "enter a price to see how many pounds you'd need", which is the more honest question for a trip not yet taken.
- **Break-even price does not need the user's price.** It needs catch, fuel and crew.
- **Crew pay under a day rate is independent of the catch.** It is shown only in the breakdown.
- **Not in the module but derivable.** Under share-after-expenses with all costs shared, net is zero exactly when `R(1−τ) − uQ = D`, which does not depend on the share *s*. So "your catch covers up to $X of fuel and shared costs" is computable **before** a crew share is chosen. Proposed addition: `costHeadroomUsd` and `beforeCrewUsd` (gross − fees − trip cost). A further proposal is to make `missing` field-level (`fuel.cruiseKt`, `crew.pct`…) so the UI can name and focus the exact field.

## 3. Assumptions and reference data, mixed

| Value | Where it comes from | How it enters | Problem |
|---|---|---|---|
| Steaming 7.8 gal/h, salmon trolling 3.6 gal/h | NOAA NWFSC-121 Table 31, a **2008** fixed-gear survey (the preset's own note: "2008 was a year of very little salmon trolling") | **Prefilled** into the field with basis `preset`; "Starting value, edit freely" appears below the field, inside the folded worksheet | It looks like the user's number. It is the only reference that is prefilled, which is inconsistent with "references beside inputs, never pre-filled" (v6 ECON) for price and diesel |
| Crew share 28.1 / 30.7 % (crab, groundfish) | NWFSC-121 Tables 32/33 | Prefilled for those fisheries, **null for salmon** | The same model select shows a share for one fishery and a blank for another, with no explanation of why |
| Diesel $8.242/gal | EIA weekly California **on-road retail including road taxes**, week of Oct 19 | Beside the field, "Use $8.242" button, basis `eia` | Handled well. The label "In use as your starting assumption (basis: EIA)" is clear. The fact that it is not the dock price is behind "About this reference" |
| Dock price | CDFW By The Numbers (species-exact only) | Salmon has none: "No species-level reference. Ask your buyer." | Correct, though the empty state is only a small grey line |
| Blank cost lines, blank fees | — | Counted as $0 | Said ("a blank line is simply not counted") only in code comments; the UI does not say it |
| Steaming leg when distance is blank | — | Counted as 0 gal | Silent (trace step 4) |
| Crew model "Share after expenses" | Code default | Shown as if chosen | The user never chose it |

## 4. Regulatory context: how the page could imply authorization

- **Commercial reader, salmon.** The glance reads "**Ocean Salmon: CDFW's sport list only.** Commercial opener: see CDFW commercial / in-season page". This is deliberate (v6 P0 2: never show the sport date as the commercial opener). The effect is that the *closed* mark disappears for commercial readers, while the recreational reader sees "Ocean Salmon is on CDFW's closed list". "Sport list only" is ambiguous: it can read as "salmon appears only on the sport list". Nowhere does the page say plainly that **Neritic holds no commercial salmon season**.
- **The economics panel takes no account of season or audience.** It shows "+$2,087 net to boat" for salmon next to a closed sport season and an unknown commercial status. Its only caveat is "Scenario from your inputs, not a forecast", which covers forecasting but not authorization. On phones, a fixed result bar ("Trip · Net +$2,087 · Cost · Break-even") follows the reader across the whole briefing with no caveat at all.
- **Recreational readers get a commercial calculator.** `Worksheet` destructures `audience` but never uses it. A recreational salmon angler sees "Starting values: Salmon troll" (a commercial gear preset), "Your price, $/lb", crew share models and "Net to boat". That implies sport-caught fish are sold. The owner should confirm the CDFW rule on selling sport-caught fish before copy is written; the design removes price and net for recreational trips either way.
- **The notice wording is mostly sound.** Notices are labelled not verified, a neutral rule is used when no notice names the species, and "No notice in Neritic's copy names these waters for salmon · Not known". The remaining risk is the glance, where "none name salmon" sits beside a number. Read quickly, it can feel like clearance.
- **Commercial and recreational differ in the glance and the Seasons fold only.** Conditions, limits, "What matters now", economics and checks are identical.

## 5. Workflow audit of the whole page (first-time user, 1440 and 390)

**What they see first (1440).** Left column: a header with three selects, then "Conditions" (wind and seas, with waves and water "not measured nearby" as two dash tiles), a tide chart, "Your limits: not set · Set", the official line, "Remembered on this device / Keep a dated copy / Forget", "What matters now" (3 statements + "1 more"), "5 things not known", seven folds (Sea, Seasons, Before you go, Evidence, Official notices, What is not known, On the map), Explore and Research links, and two paragraphs of caveats. Right column: the economics panel, sticky, **with its own inner scroll** (`max-h: 100dvh; overflow-y: auto`), which makes a scroll area inside the page.

**What they must do.** The page has no stated task. Economics is the only interactive instrument, and its required inputs are folded away. Limits are a small "Set" link. Nothing on the page states the decision being made: whether this trip is worth making, within my limits, given what officials say.

**Where they get lost.**
- Three dashes plus a "Needs" sentence, with no route to the fields (section 1).
- **51 `<details>` elements** on the page: disclosures nested two deep in economics ("Full worksheet" → "About this reference"), plus "Breakdown" placed above the inputs it summarises.
- On a phone (3,255 px tall) the economics panel starts at y ≈ 980. The "Full worksheet" toggle is at y ≈ 1,770, two screens into the panel, and it holds the two required inputs.
- The map, which the brief calls the route to geographic evidence, is the last fold ("On the map") plus a footer link.

**What is repeated.** The notice count appears three times: the top bar "3 here", the glance line, and the "Official notices" fold summary. NWS appears in the conditions tile, the "Marine forecast (NWS), Tonight" link and the Sea fold. Device storage is mentioned in "Kept on this device only", "Remembered on this device", "Your numbers stay…" and the footer. Season appears in the glance and the Seasons fold. Unknowns appear as "5 things not known" and in the "What is not known" fold.

**What is buried.** Crew pay and fuel quantity (both required); gross; sensitivity (absent until complete, with no placeholder); scenarios and compare (at the bottom of Full worksheet); the source of the burn-rate presets; and the statement that Neritic has no commercial salmon season.

## 6. What the v6 and v7 history decided, and what this audit keeps

The audit keeps these decisions: nothing pre-filled for catch or price; references beside inputs with a visible "use" action; device-only storage; all three crew models; scenarios, compare and the share link; the copy "Scenario from your inputs, not a forecast" and the banned words (safe, go/no-go, good day, hotspot, expected catch, price forecast); and no trip score.

It revisits one: v7 moved the v6 "At a glance as rows: …trip cost; break-even; missing/uncertain" into a quick block holding four fields. The missing/uncertain row was lost, and so was the route to the remaining required inputs.
