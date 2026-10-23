# My Coast as a decision workspace (v10 proposal)

Inputs: `my-coast-economics-audit.md` (what is broken and why) and `decision-ux-references.md` (GFW, Copernicus Marine, NN/g, calculator references).
Prototype: `scratchpad/v10-econ/prototype/index.html`. It is standalone and opens from `file://`. Its states are `?state=missing|partial|complete` and `&for=recreational`, and a state bar sits bottom right.
Screens are at 1440×900 and 390×844, in `scratchpad/v10-econ/proto-{missing,partial,complete,recreational}-{1440,390}.png`, with `-fold` crops at 1440.
The prototype's arithmetic is the app's own `src/lib/tripEconomics.ts`, compiled by `tsc` into `prototype/tripEconomics.js` and `tripEconomics.global.js` with its sha256 noted in the header. The page adds no maths of its own beyond two display subtractions, flagged below as proposed module outputs.
The visual direction is owned elsewhere. This document covers structure, flow and the economics graphic only.

## 1. The decision

The page answers one question for one trip: **"Is this trip worth making, within my limits, given what the agencies say, and what don't I know yet?"** It does not answer it *for* the user. There is no score, no go/no-go and no verdict. Every element either supplies one of the four inputs to that question or says that it is missing.

## 2. Structure

```
Trip frame   Santa Cruz · [Commercial | Recreational] · Salmon ▾ · Thu Oct 22 ▾
             one line: research preview · official items are unverified copies · nothing here says you may or may not fish
──────────────────────────────────────────────────────────────────────────────────────────
DECISION BOARD (4 cards, one line deep, each links to where it is resolved)
 Official context      │ Sea vs your limits     │ Trip economics          │ Still open (n)
 unverified copy chip  │ NWS forecast chip      │ hypothetical chip       │ ○ commercial season: check CDFW
 sport list: closed    │ NW 10–15 → 15–25 kt    │ gross $3,600 · cost $618│ ○ your limits
 Neritic holds no      │ seas 3–4 ft · SCA      │ net needs: crew pay ›   │ ✓ catch and price
 commercial season ›   │ limits: set inline     │ not permission to fish  │ ○ fuel: cruising speed ›
 ☐ I checked (my note) │ max 25 kt: over 20 kt  │                         │ ○ crew pay ›
 Evidence on the map:  Santa Cruz in Explore ↗ · 3 notices' limits ↗ · Wharf sampling 2.2 km ↗   (keep place/species/date)
──────────────────────────────────────────────────────────────────────────────────────────
TRIP ECONOMICS (open, not folded)
 Inputs as a ledger, in the order money moves      │ Results (sticky at ≥ 860 px; first on phones)
 1 What the catch would bring  lb · $/lb            │ Net to boat │ Trip cost │ Break-even catch
 2 Fuel  [from the trip | I know the gallons]       │ waterfall: gross − fuel − costs − crew = net
        nm · kt · steam burn · hours · fish burn    │ break-even strip: 0 … ▮ break-even … ● your lb
        references as rows: [Use 7.8] [Use 3.6]     │ sensitivity 3×3 + fuel ±20 %
        fuel price  [Use EIA $8.242]                │ saved scenarios: chips + differing-inputs compare
 3 Other trip costs (optional; blanks not counted)  │
 4 Crew pay  (4 cards: none · after expenses ·      │
             of gross · day rate) + its parameters   │
──────────────────────────────────────────────────────────────────────────────────────────
MORE FOR THIS TRIP (one level of disclosure, never nested)
 Sea forecast and tides · Seasons · Official notices (3) · Evidence (8) · Before you go · What is not known (5)
```

What changes from today:
- **Opening state:** the board replaces "Conditions" + glance line + "What matters now" + "5 things not known" at the top. "What matters now" moves into the Evidence and Notices folds; it repeated the notice and measurement lines. The board card for each topic is the *only* summary of it. Its fold holds the detail. Nothing is said twice.
- **"Still open"** is new. It lists, across the whole page, the facts the user has not supplied yet: commercial season checked, limits, and each economics input, named at field level. Each item is a button that focuses the field. It takes the "missing / uncertain" row from v6, which v7 lost.
- **Evidence route:** the map links sit directly under the board, not in the last fold or the footer. GFW and Copernicus both keep the map one click from every number (see the references doc). Each link keeps `place`, `for`, `species` and `date`.
- **Economics is no longer a side column with its own scroll.** At 1440 it is a full-width two-column block: inputs on the left, results on the right in a sticky column. On phones the results come first, with a compact sticky summary (unchanged from today's bar, but carrying the "hypothetical" chip).
- **Disclosure** is one level only: the six "More" folds. The economics panel has no `<details>` at all. Reference provenance is on the value (a chip and a one-line source), not under "About this reference".

## 3. Official context: concise, contextual, never a verdict

- The card always states **which list** it quotes and its read date: "CDFW's *sport* list shows Ocean salmon closed (read Oct 20)". For a **commercial** reader the closed mark is no longer hidden. The current "sport list only" wording drops it. The card then says plainly: "Neritic holds no copy of the commercial salmon season. Check before you plan", followed by the CDFW commercial page, the in-season page and the NOAA hotline. This keeps the v6 P0 2 rule (the sport date is never shown as the commercial opener) and removes the ambiguity.
- "☐ I checked the commercial status myself (your own note)" works the same way as the Before-you-go ticks. It is the user's note, stored on the device, and never a Neritic verification. It only clears the "Still open" item.
- Notices: "3 tied to these waters; none name salmon; 1 status unclear". The card has the unverified chip and never says "clear" or "no closures".

## 4. Sea vs your limits

- The forecast is labelled as a forecast (NWS, period). Measurements stay labelled as measurements.
- Limits are set **inline in the card** (two fields), not through a "Set" link. Once set, the card reads "Forecast max wind 25 kt: over your 20 kt. Seas 4 ft: within your 6 ft. Your limits compared with NWS text. Not a go/no-go." It uses `marineMaxima` and `overLimits` from `src/lib/trip.ts` unchanged.

## 5. Economics interaction model

**Principle: every result appears the moment its own inputs exist. Every result that cannot appear names exactly what it needs, as a control that takes you to that field.**

1. **Ledger order = money order.** The sections are revenue, then fuel, then other costs, then crew. Each step has a status ("Complete" / "Needs cruising speed") and a one-line worked sum underneath, using the module's own outputs: "Gross **$3,600** = 400 lb × $9.00/lb". "Fuel **75 gal** (40 nm ÷ 8 kt × 7.8 gal/h + 10 h × 3.6 gal/h) × $8.24 = **$618**". "Crew **$865** = 30 % × $2,882 (gross − fuel and shared costs)".
2. **Missing inputs at the point of need.**
   - Fuel offers a two-way choice: *Work it out from the trip* or *I know the gallons*. The gallons route is one field, for the skipper who knows his usual burn.
   - When nothing is entered, the gap is one item, "fuel used (gallons, or the trip's distance, speed and burn)". Once any field is entered, the gaps become field-level ("cruising speed"), and blank required fields get a quiet outline (not error red).
   - If only hours are given, the sum line says "Only fishing hours are counted: add the round trip to include steaming". The module's silent zero-steaming assumption is now stated.
   - Crew pay is **four cards with no default**: No hired crew, Share after expenses, Share of gross, Day rate. Each card says in one line what it means. Choosing one reveals only its parameters. Until one is chosen the step reads "Choose one: needed for net" and the explanation says "Neritic has no sourced crew share for salmon troll, so nothing is assumed."
3. **Partial results.**
   - Result figures that cannot compute show "Needs *fuel used* · *how crew is paid*" as links, not "—".
   - The waterfall draws solid bars for known terms and dashed bars, labelled with what they wait for, for unknown terms. Gross is visible as soon as catch and price are in.
   - The board card shows "gross $3,600 · cost $618". Net appears once crew is chosen.
4. **References vs assumptions.**
   - **No reference is prefilled.** That covers the NWFSC burn rates as well as diesel and price, and it changes today's prefilled burn rates (decision D1).
   - Each reference sits in a tinted row under the field with its source, period and caveat, plus "Use 7.8 for steaming".
   - Every filled field carries a chip: **Yours** or **Reference: EIA California diesel, week of Oct 19, 2026**. Typing over a reference flips the chip to Yours.
   - Blank optional costs say "Blank: not counted", and blank fees say "Blank: no fees counted".
   - Salmon has no CDFW price reference, so the price field says so and points to the buyer.
5. **Real time.** Computation runs on every keystroke, using `parseField` for parsing, error text and clamping. There is no Calculate button. Results sit in an `aria-live="polite"` region.
6. **Break-even is the threshold, drawn.**
   - A strip runs from 0 lb, through a shaded loss zone to the break-even mark, into a gain zone, with the user's catch as a dot: "your 400 lb (320 lb above)".
   - Below it: "At 400 lb, net reaches $0 at $1.80/lb (you entered $9.00). Fuel would have to reach $46.67/gal."
   - When the strip cannot draw, it says what it waits for.
7. **Compact sensitivity.** This is the module's 3×3 grid: price ±20 % by catch ±30 %, with the centre outlined and losses in the negative colour, plus fuel ±20 %. Before net exists it is a one-line placeholder rather than absent.
8. **Saved scenarios.** "Save this scenario" adds a named chip. Picking a chip shows a compare table with only the inputs that differ, then Trip cost, Net and Break-even for both. It is device-only, as today. The share link and import/export keep their current behaviour (not prototyped).
9. **Hypothetical framing.**
   - A dashed note heads the inputs: "Hypothetical scenario: your numbers worked through. Not a catch, price or profit forecast, and not a statement that salmon may be fished on this date."
   - The results header and the board card carry a "Hypothetical" chip.
   - The board card adds "not permission to fish".
   - There is no catch or price prediction anywhere.
10. **Commercial vs recreational.**
    - Recreational switches the panel to *Cost of the trip*: fuel and other costs, plus an optional split between anglers.
    - It shows no price, crew, net or gross, and the Salmon troll preset never appears. This uses `computeTrip` with no catch lines, which already returns `tripCostUsd`.
    - The page states why with a neutral line; the exact wording should cite the CDFW sale rule once the owner has confirmed it.

## 6. What the prototype shows (Santa Cruz, salmon, commercial)

| State | Inputs | What appears |
|---|---|---|
| Owner's state (`missing`) | 400 lb, $9/lb, fuel price via "Use EIA" | Gross $3,600 (sum line, waterfall bar, board card). Net, Cost and Break-even each say "Needs fuel used · how crew is paid" as links. "Still open: 4": commercial status, limits, fuel, crew pay. |
| Partial | + 40 nm, 8 kt, burn references accepted, 10 h; limits 20 kt / 6 ft | Fuel 75 gal, $618. Trip cost $618. Waterfall through fuel and other costs; crew dashed "needs how crew is paid". Board: "gross $3,600 · cost $618"; wind "over your 20 kt". |
| Complete | + ice $60, food $40, share after expenses 30 % | Net +$2,017, cost $718, break-even 80 lb (strip), break-even price $1.80/lb, fuel break-even $46.67/gal, 3×3 grid, compare vs a saved "Slow day, 250 lb" (net +$1,072). |
| Recreational | the same fuel inputs | Trip cost $618, fuel 75 gal, optional per-angler split; no price, crew or net. |

A Playwright interaction check from the owner's state: entering 60 gallons immediately shows "gross $3,600 · cost $495" with "needs how crew is paid". Choosing "No hired crew" gives "+$3,105 net · break-even 55 lb · cost $495". No console errors and no horizontal overflow at 390.

## 7. Proposed module additions (none made; `tripEconomics.ts` untouched)

- `missingFields`: field-level keys alongside today's `missing` (for example `fuel.cruiseKt`, `crew.kind`, `crew.pct`). The prototype derives these in its UI from the blanks, which is fine for a prototype. The app should take them from the module so the logic has one owner.
- `beforeCrewUsd = gross − fees − tripCost`. The prototype computes the model-A crew pool for its explanation line by subtracting module outputs. It should come from the module.
- `costHeadroomUsd`: the fuel and shared costs the entered catch can carry before net reaches $0. Under share-after-expenses with shared costs it does not depend on the crew share, so it can be shown before crew is chosen.
- `assumptions[]`: the module's implicit zeros ("steaming not counted", "blank cost lines not counted", "no fees") returned as data, so the UI never has to restate them by hand.
- Crew `kind: "none"`. Today the prototype maps "No hired crew" to after-expenses at 0 %, which is arithmetically identical.

## 8. Decisions for the owner

- **D1:** stop prefilling the NWFSC burn-rate presets and offer them as one-tap references, like diesel. This is recommended: the prefilled 7.8 / 3.6 are the main reason "fuel" looks done when it is not. The alternative is to keep the prefill and add the reference chip.
- **D2:** give the crew model no default (recommended), versus keeping "Share after expenses" preselected with only its % blank.
- **D3:** show the closed sport mark to commercial readers alongside "Neritic holds no commercial season" (recommended), versus today's "sport list only".
- **D4:** recreational economics as cost-only (recommended), versus hiding economics entirely for recreational trips.
- **D5:** the "I checked the commercial status myself" note. Do we want any user-ticked item on the official card at all, or should it live only in Before you go?

## 9. Not done / limits

- The prototype is structural: neutral styling, static Santa Cruz data, no share link, no import/export, and no vessel save.
- The "More" folds are placeholders for today's content.
- Map links are inert.
- The phone layout puts results before inputs. The sticky phone summary bar from today should be kept and carry the "Hypothetical" chip; it is not prototyped.
