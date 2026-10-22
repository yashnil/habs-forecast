# Decision UX references for the My Coast decision workspace

Research date: 2026-10-22. Scope: how strong ocean-data and calculator products handle inputs, partial results, provenance, feedback and map-to-evidence navigation, and what Neritic's My Coast trip briefing should adapt.

Method: every claim below comes from a page that was fetched and read on this date. Where a page did not state a behaviour, the doc says so instead of assuming it. No product was used interactively; these are documented behaviours, not hands-on tests.

## Neritic baseline (as verified by the lead)

- My Coast is two long columns of text and controls, with several nested `<details>` disclosures ("Full worksheet", "Breakdown and sensitivity", "About this reference").
- Trip economics shows Net / Trip cost / Break-even as "—" even when catch, price and fuel price are filled. Fuel quantity (distance/speed/hours or gallons) and crew share are required, but those inputs sit in the collapsed "Full worksheet". The only explanation is a plain-text "Needs: fuel use (...); crew share." line.
- Gross (catch x price) can be computed from what the user typed, but it is hidden in a collapsed breakdown.
- Reference data (EIA weekly diesel price, CDFW historical dock price, 2008 NOAA NWFSC fuel-burn presets) sits beside user inputs with "Use $X" buttons. Preset burn rates are prefilled into input fields, marked only by a small "Starting value" note.
- Map evidence lives on a separate page.

---

## 1. Global Fishing Watch (map and user guide)

Sources read: https://globalfishingwatch.org/ and https://globalfishingwatch.org/user-guide/ (FAQ index https://globalfishingwatch.org/faqs/ lists only question titles; the answers sit on separate pages that were not opened).

**What it does exceptionally well**
- One time range drives the whole workspace. The guide's "Timebar" and "Selecting a time range" sections describe a date picker plus draggable start and end bars. Detail level changes with range length: daily under three months, monthly above that, yearly above six years. Animation plays activity over the chosen window.
- Every layer can be inspected in place. An "i" icon beside each layer opens "its methodology and data source". Per-layer filters (source, flag, gear type) live on the layer row.
- The map leads straight to evidence. Clicking a grid cell in an activity layer opens a pop-up listing the vessels with the most activity in the selected time range. "See vessel" pins the track to the sidebar, and the vessel profile has REGISTRY, AIS, ACTIVITY, AREAS and Related Vessels tabs. Clicking an encounter, loitering event or port visit shows its details.
- Areas lead to reports. Users can draw an area or pick a reference area (EEZ, MPA, RFMO). A graph icon beside an area, or in a cell pop-up, opens a report with an "Evolution" graph, "Before/After", "Period Comparison", "Buffer Area" and "Download CSV".
- Caveats are stated. The guide says "Events are estimated using machine learning", notes that VMS data is owned by states and shared with varying availability, and flags night-light false positives.

**Why it feels polished**
- Area, time and layer form one model. Changing any of them reframes the map, the pop-ups and the reports together, so the user never has to work out which date a number belongs to.
- Provenance is one click from the thing it describes (the "i" next to the layer), not on a separate methodology page.

**What Neritic currently does worse**
- Map evidence lives on another page, so a reading in the briefing (a buoy value, an advisory area) cannot be opened in its spatial context, and the map cannot answer "what does this mean for my trip".
- Reference-data provenance sits in collapsed "About this reference" blocks away from the number it qualifies, not in an "i" beside it.
- No single time frame visibly scopes the briefing. Forecast, buoy, tide and notice timestamps each live in their own block.

**Principle to adapt**
- Every evidence item in the briefing (buoy, tide station, notice area, forecast zone) should link to the map, focused on that feature and time, and the map pop-up should link back to the briefing row. Provenance gets an inline "i" or source chip on the item itself.
- Use one visible trip window (for example "Thu 05:00 to 17:00") that scopes the forecast, tide and limit comparisons.

**What not to copy**
- The analysis depth: bivariate mode, buffer areas, period comparison, CSV export, custom polygon upload. These are tools for analysts. A fisherman's trip briefing needs a yes/no-with-reasons view, not an analysis suite.
- The timebar as the main control. GFW's range goes back to 2012. Neritic's window is hours to days, and showing observations on a scrubbable timeline next to forecasts risks blurring the two, which the project rules forbid.

---

## 2. Copernicus Marine (portal, MyOcean viewers, product pages)

Sources read: https://marine.copernicus.eu/ ; https://marine.copernicus.eu/access-data/ocean-visualisation-tools ; https://help.marine.copernicus.eu/en/articles/9418874-introduction-to-myocean-light-viewer ; https://help.marine.copernicus.eu/en/articles/4794675-introduction-to-myocean-viewer ; https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description. The feature-comparison table on the visualisation-tools page did not come through cleanly, so this doc makes no claim about which viewer has which feature.

**What it does exceptionally well**
- Audiences are tiered explicitly. MyOcean Learn ("Understand key variables", beginner), MyOcean Light ("Access key variables", intermediate) and MyOcean Pro ("Access full catalogue", 300+ products) are separate entry points, not one interface with everything hidden in it.
- MyOcean Light removes a decision from the user: "you don't have to select a Copernicus Marine data product (and you can't)." The viewer picks the product from the selected date and zoom, and switches to a "higher-resolution regional product" when one exists. Provenance is still one click away: "the link above the legend" opens the product page, and dataset IDs are tabulated. Names such as ANALYSISFORECAST, MULTIYEAR and REANALYSIS show the product type.
- Product pages give provenance as labelled fields: "Product ID", "Source: Numerical models", "Processing level: Level 4", "Temporal extent", "Temporal resolution", "Update frequency: Daily – 08:00 UTC", "Spatial resolution", DOI, plus "User Manual" and "Quality Information Document".
- A point click gives evidence right away. In Light, clicking the map opens a graph window with the point's coordinates, the value at the selected time and depth, a time series and a depth profile, exportable as CSV, netCDF or SVG.

**Why it feels polished**
- Every number answers "what, where, when, from which product": the selected time applies to all layers, the depth axis appears only for variables that have depth, and the legend links to its product.
- Simplicity comes from choosing for the user, not from hiding controls in nested panels. The expert path is a different tool.
- Update frequency and source type ("Numerical models") are stated plainly, so model output is not mistaken for observation.

**What Neritic currently does worse**
- Reference values (EIA diesel, CDFW dock price, 2008 NWFSC burn presets) do not carry a compact, always-visible "source, as-of date, type" label. Provenance sits in collapsed "About this reference" text, and prefilled presets look like user input apart from a small "Starting value" note.
- The 2008 burn survey's age is a provenance fact that matters (fleet and engines have changed since). Copernicus would state the temporal extent as a field. Neritic hides it.

**Principle to adapt**
- Give each reference value a one-line provenance label next to the value: source, as-of date or period, and kind (weekly survey average, historical dock price, 2008 fleet survey preset). This mirrors Copernicus's "Source / Temporal extent / Update frequency" fields at a fraction of the size.
- Where Neritic can choose the right reference for the user (port to dock-price series, gear to burn preset), choose it, show it as a reference suggestion, and keep the product link one click away, as Light does.

**What not to copy**
- Catalogue scale and IDs. Product codes like GLOBAL_ANALYSISFORECAST_PHY_001_024 mean nothing to a skipper. Use plain names, with IDs only behind the source link.
- Three separate viewers. Neritic is one product for one audience. The lesson is to drop expert features from the briefing, not to build a second app.

---

## 3. Nielsen Norman Group: disclosure, feedback, forms

Sources read: https://www.nngroup.com/articles/progressive-disclosure/ ; https://www.nngroup.com/articles/response-times-3-important-limits/ ; https://www.nngroup.com/articles/visibility-system-status/ ; https://www.nngroup.com/articles/errors-forms-design-guidelines/ ; https://www.nngroup.com/articles/web-form-design/. A search for an NN/g article dedicated to inline validation found none, so inline-validation guidance here comes from the errors-in-forms article.

**What it says exceptionally well**
- Progressive disclosure: "You have to disclose everything that users frequently need up front." "Designs that go beyond 2 disclosure levels typically have low usability." If 3 or more levels seem necessary, "consider simplifying your design." The trigger label must set "clear expectations for what users will find". Staged disclosure "becomes problematic when steps depend on each other".
- Response times: within 0.1 s, "no special feedback is necessary except to display the result". Past 1 s, show that work is in progress. Past 10 s, show a percent-done indicator and let users interrupt.
- Visibility of system status: "Only by knowing what the current system status is can you change it." "A lack of information often equates to a lack of control." The article also says to show unavailable items with an explanation rather than silently removing them.
- Errors in forms: put messages next to the field ("minimizes working-memory load"). A summary "shouldn't be used as the only form of error indication." Avoid tooltip-only errors. Messages should be "explicit, human-readable, polite, precise, and give constructive advice". Validate inline, but only after the user leaves the field.
- Web form design: state input rules up front and avoid hidden rules users learn from an error message. Keep labels next to their fields. Group related fields. Avoid placeholder text. Keep optional fields to 1 or 2 and label them.

**What Neritic currently does worse**
- The economics calculator breaks the "frequently needed up front" rule. Fuel quantity and crew share are required for the headline result but sit in a collapsed "Full worksheet". That is a hidden rule in the form-design article's terms.
- The dashes for Net / Trip cost / Break-even are a status the user cannot act on. The "Needs: ..." line is a detached summary, the pattern NN/g says must not be the only indication, and it does not move the user to the missing field.
- Gross is computable but hidden, which goes against "display the result" and against showing what is available with an explanation rather than withholding it.
- Nesting ("Full worksheet" containing "Breakdown and sensitivity" and "About this reference") reaches or passes the 2-level limit. "Full worksheet" does not tell the user that required inputs are inside it.
- Prefilled preset values act like placeholders or defaults the user did not choose. The form-design article warns about placeholders, and the result does not state its assumptions.

**Principle to adapt**
- Required inputs live on the first level. Only refinements (crew share split detail, sensitivity) are disclosed, at most one level deep, and the trigger names what is inside it.
- Compute on every input change (well within 0.1 s for arithmetic). Show every figure that can be computed now, and put each missing input's explanation next to the result it blocks, as a control that focuses that field.

**What not to copy**
- Red error styling for fields the user simply has not filled yet. NN/g's timing rule (wait until the field is finished) and its tone guidance point to a neutral "add fuel use to see trip cost" prompt, not an error.
- Staged wizards. Trip-economics inputs depend on each other (speed, hours, distance, burn), which is the case where NN/g says staging fails.

---

## 4. Calculator reference: NerdWallet and Calculator.net mortgage calculators

Sources read: https://www.nerdwallet.com/mortgages/mortgage-calculator (first 100,000 of about 111,000 characters) and https://www.calculator.net/mortgage-calculator.html. Omni's break-even calculator (https://www.omnicalculator.com/finance/break-even) was also read, but its page text did not confirm live recalculation, solving for missing inputs, or missing-input behaviour, so it is not used as evidence.

**What they do well (verified)**
- NerdWallet labels the source of an assumption at the point of input. The interest-rate control offers "National Avg 7.10%", "Enter own" and "Best on NW 6.89%". The result restates the assumption it used ("$2,114 at 6.89% · 30-year fixed"), lists components (principal and interest $1,579, property taxes $375, insurance $160), and labels outputs as estimates ("Calculated payments and savings are estimates").
- NerdWallet shows a field only when it applies: the PMI field "appears only when the down payment is under 20%".
- Calculator.net shows the result and its decomposition together: a headline monthly payment, a Monthly/Total breakdown table, and a percentage split (72% / 13% / 4% / 11%).

**Why it feels polished**
- The headline number never stands alone. It carries its key assumption and its parts, so users can see why it is what it is.
- A reference value is a choice the user makes visibly (a chip), not a value that silently fills the field.

**What Neritic currently does worse**
- "Use $X" buttons sit beside fields but are not presented as a labelled choice between reference and own value. Prefilled burn presets blur the line between reference and user input.
- The Net / Trip cost / Break-even line does not restate the fuel price, burn rate and share it used, and the breakdown sits behind a disclosure.

**Principle to adapt**
- A reference value is offered as a labelled choice ("EIA weekly CA diesel, week of ...", "Enter own"). Once applied, the field shows a small "reference" tag until the user edits it, after which it reads as "your value". The result line repeats the assumptions it used.

**What not to copy**
- Calculator.net's explicit "Calculate" button ("Modify the values and click the calculate button"). It adds a step and a stale state that live arithmetic does not need.
- NerdWallet's lender-ranked "Best on NW" rate is a commercial default. Neritic must not rank or prefer a reference value for any reason other than relevance, and should never imply a dock price or fuel price is a forecast or a quote.

---

## Principles for the decision workspace

1. **Point-of-need inputs.** Every input the headline result needs (catch, price, fuel price, fuel use, crew share) is on the first level, next to the result it feeds. Nothing required is ever behind a disclosure (NN/g: "disclose everything that users frequently need up front"; no hidden rules).
2. **Partial results, always.** Show each figure as soon as its own inputs exist: gross as soon as catch and price exist, fuel cost as soon as fuel use and price exist, net and break-even when the rest arrive. Never blank the whole line because one term is missing.
3. **Explicit missing-input explanation, in place.** Where a figure cannot be computed, say what it needs in that figure's slot ("Add fuel use to see trip cost") as a control that focuses the field. Use a neutral tone, not error red, and never rely on a detached "Needs:" summary alone (NN/g errors-in-forms).
4. **Provenance labels separate user assumption from reference data.** Each reference value carries source, as-of date and kind (EIA weekly average; CDFW historical dock price; NWFSC 2008 fleet-survey preset). Applying a reference is a visible choice, and a field shows "reference" until the user edits it, then "your value". Do not silently prefill. Result lines restate the assumptions they used (Copernicus product fields; NerdWallet rate chips).
5. **Real-time feedback.** Recompute on every input change with no Calculate button. Arithmetic fits the 0.1 s "just display the result" limit. Show the as-of time of any fetched data so users know the system state (NN/g response times; visibility of system status).
6. **Map-to-evidence navigation, both ways.** Each buoy, tide station, notice area and forecast zone in the briefing opens the map focused on that feature and time, and each map pop-up links back to its briefing row with its source and timestamp (GFW cell pop-ups to vessel and event detail; Copernicus point click to value and time series). Observations stay labelled as observations and never as forecasts; chlorophyll is never presented as toxin; notices stay labelled as unverified copies.
7. **Progressive disclosure, at most one level of nesting.** One disclosure per block, for refinements only (sensitivity rows, methodology text). Its label names what is inside ("Sensitivity: price and fuel +/-10%"). No disclosure inside a disclosure (NN/g: more than 2 levels "typically have low usability").
8. **Provenance inline, not in an appendix.** Put an "i" or source chip on the value it qualifies, not an "About this reference" section (GFW layer "i" icon; Copernicus "link above the legend").
9. **One visible trip frame.** A single port, fishery and trip-window header scopes the forecast, tide, limits and economics, so each number answers "for which trip, when" (GFW's single time range; Copernicus's shared date-time).
10. **Choose for the user, and show the choice.** Pick the relevant reference (port's dock-price series, gear's burn preset) automatically as a suggestion, and keep the source link and "Enter own" one click away (MyOcean Light's automatic product selection with a visible product link).
