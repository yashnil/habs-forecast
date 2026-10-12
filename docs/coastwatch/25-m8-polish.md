# 25 — M8: final product polish and release readiness

**Status:** draft PR for review; not merged, not deployed publicly. Web only: no pipeline, data or forecast change. The Research Preview is untouched.

**Method:** the integrated site (M5 Ocean Map, M6 Bloom, M7 Fisheries, #28 coastline) was reviewed in a real browser on production data. The private preview requires a Vercel login and its bypass token could not be created in this session, so every screenshot and measurement comes from a local production build of the same commits reading the same production data (`https://yashnil.github.io/habs-forecast/v1`). The baseline is `main` at 9802a4a7.

## 1. What the audit found (most important first)

| # | Problem | Where | Evidence |
|---|---|---|---|
| 1 | **Link colours were broken site-wide.** An unlayered `a { color: inherit }` outranked every Tailwind class, so links meant to be teal (and white text on dark buttons) rendered as body text | every page | found when the About page's "Read the paper" button rendered blank |
| 2 | **No answer to "what is this?".** No About page, nothing on the map for a first-time visitor, and the research was a paragraph on Data & sources | site-wide | journey 1, journey 6 |
| 3 | **Official status said several times over.** Status line, masthead pill, a page strip with its own "transcribed, not checked" sentence, and a second red box in the port inspector | map inspector, Bloom, Fisheries | port view: "Not verified" 4× on the first screen at 1440, 5× at 390 |
| 4 | **Phone Bloom showed no measurement on the first screen.** A 165 px notices box, a station card that repeated the station name, the coordinates and helper text came first | Bloom 390 | [before](m8/m390-bloom.jpg) |
| 5 | **Fisheries on a phone: 4,683 px with no way to move through it.** Reading notes, tier notes and the limits section were all expanded | Fisheries 390 | [before](m8/m390-fisheries-full.jpg) |
| 6 | **Type scale drifted.** About 120 interface sizes of 10–12.5 px (map panels, notice cards, badges), against the spec's 13 px minimum | map panels, cards | |
| 7 | **Map labels and port markers waited on a third-party font server** (MapLibre holds a source's tiles until its glyphs arrive) | Ocean Map, Bloom map | #28 limitation |
| 8 | **Phone: "All California" closed the Places sheet**, so changing region took two extra taps | Ocean Map 390 | journey 5 |
| 9 | **Keyboard:** no skip link (nine Tab stops before content on every page); the map page had no h1; Data & sources had two "Source status" headings; moderate landmark duplicates | all pages | |
| 10 | **Layout shift:** the status line appeared after hydration and pushed every page down 35 px (CLS up to 0.07) | all pages | |
| 11 | "Issued … (inferred)" with no explanation | map, port panel, sources | journey 1 |

## 2. What changed

Shared components first; page-specific changes only where the problem was page-specific.

- **Links and type.** The anchor rule is in the base layer, so link colours work. Links inside sentences carry a light underline (never colour alone). Interface text follows the spec: 13 px reading text, 12 px uppercase labels and badges; chart axes and map label sizes unchanged.
- **One notice treatment.** The status line under the masthead explains the state once (not checked by a person, may be incomplete, CDFW and CDPH links). Below it, notices appear once per context with one "Not verified" badge: a compact row on Bloom and Fisheries that always keeps "A missing notice does not mean an area is open or seafood is safe."; the port inspector keeps its badge and loses the second box.
- **About CoastWatch** (`/about`): what it is; the four kinds of information (official, model, measured, historical) in their reserved colours, each with where to find it; what CoastWatch cannot tell you; the publication with its same-pipeline result and what it does not cover; data providers and methods; review status. Reached from the masthead, a fifth phone tab, a footer on every reading page, Data & sources and the first-visit card.
- **First visit on the map:** a two-sentence introduction under the place chip (official information stays above it) with the Model, Measured and Official chips, About and "Got it"; shown once per device.
- **Phone layouts.** Bloom: the picker is a "Change" button in the station header, the notices row and header are compact, coordinates move to the provenance line; the first measured value is on the first screen. Fisheries: section links (By year, By species, Can and cannot, Ports, Methods), "How to read this chart" one tap away, "They do not show" open with "They show" folded below.
- **Map reliability.** Label glyphs (Noto Sans Regular and Italic, Latin ranges) are served by the site; any other range still comes from OpenFreeMap. Port markers now draw while the font server is slow (tested with a 40 s delay).
- **Interaction and wording.** "All California" keeps the phone's Places sheet open; "(date inferred)" with "NOAA does not publish the issue date"; skip link on every page; an h1 on the map; one Source status heading; drawer and phone sheet landmarks de-duplicated.
- **Layout stability.** The status line holds its place while the browser decides it.

## 3. Before and after (`main` left, M8 right; production data)

| View | Comparison |
|---|---|
| Ocean Map, 1440×900 (first visit) | [d1440-map](m8/d1440-map.jpg) |
| Ocean Map, port selected | [d1440-map-port](m8/d1440-map-port.jpg) |
| Ocean Map, 1280×720 and 1024×768 | [d1280-map](m8/d1280-map.jpg), [t1024-map](m8/t1024-map.jpg) |
| Bloom, 1440×900 and 1280×720 | [d1440-bloom](m8/d1440-bloom.jpg), [d1280-bloom](m8/d1280-bloom.jpg) |
| Fisheries, 1440×900 and 1024×768 | [d1440-fisheries](m8/d1440-fisheries.jpg), [t1024-fisheries](m8/t1024-fisheries.jpg) |
| Data & sources → About | [d1440-about](m8/d1440-about.jpg) |
| Phone 390×844: map, port, Bloom, Fisheries | [m390-map](m8/m390-map.jpg), [m390-map-port](m8/m390-map-port.jpg), [m390-bloom](m8/m390-bloom.jpg), [m390-fisheries](m8/m390-fisheries.jpg) |
| Phone Fisheries, whole page | [m390-fisheries-full](m8/m390-fisheries-full.jpg) |

Every page was captured at 1440×900, 1280×720, 1024×768 and 390×844 (88 screenshots, before and after); none has a page error or horizontal scroll.

**Phone page heights:** Bloom 2,825 → 2,815 px with the first value on the first screen (was below it); Fisheries 4,683 → 4,470 px including the new 170 px footer, with section links; Sources 4,205 → 4,539 px (it gained the About link and the footer).

## 4. User journeys (production data, real browser)

| Journey | Result | Found and fixed |
|---|---|---|
| 1. First-time visitor, Monterey Bay | Intro → Moss Landing → forecast near the port → "Measured nearby" → Monterey Wharf: 4 clicks | "(inferred)" unexplained |
| 2. Fisher checking notices (phone) | Notices tab → drawer with CDFW, CDPH and hotlines → Places → "Morro" → Morro Bay with its notices first: 4 taps | — |
| 3. Researcher, historical toxin | Monterey Wharf → particulate domoic acid → Since 2014: 3 clicks; values table one tap away; source and station location in the provenance line | no data download (outstanding) |
| 4. Fisheries 2015–16 | 2016 → year panel with the closure record → pounds → Dungeness crab: 3 clicks | — |
| 5. Phone navigation | Places → region; Blooms → change station; Fisheries → Methods; About | "All California" closed the sheet |
| 6. Reviewer: paper, data, methods | About in the masthead → "Read the paper" (DOI) → what it does not cover → Data & sources → code and documentation | the research was not reachable from navigation |

## 5. Accessibility

- **axe** (all rules) on every page at 1440 and 390, plus the open notices drawer, an open port and the phone station picker: **0 violations** in all 16 states (before: 2 moderate landmark issues; the link fix had exposed one serious link-in-text-block issue, fixed).
- **Keyboard:** the first Tab on every page is "Skip to main content", which moves focus to the content; every focus stop checked has a visible ring; the drawer traps and returns focus (existing tests); one h1 per page.
- **Screen readers:** checked through the accessibility tree (headings, landmarks, names), not with VoiceOver or NVDA in this session. Landmarks are unique; the map has an h1; badges and counts carry text, not colour alone.
- **Reduced motion:** all animations and transitions are off under `prefers-reduced-motion` (unchanged).

## 6. Performance (median of 3; production data; phone = 390×844 with CPU slowed 4×)

| Page | Device | LCP `main` → M8 | CLS | Blocking time | Interaction |
|---|---|---|---|---|---|
| Ocean Map | desktop | 148 → 160 ms | 0.039 → 0.017 | 0 → 0 ms | 48 → 48 ms |
| Ocean Map | phone | 528 → 540 ms | 0.023 → 0.036 | 138 → 125 ms | 72 → 72 ms |
| Bloom | desktop | 104 → 104 ms | 0.074 → 0.027 | 0 → 0 ms | 40 → 40 ms |
| Bloom | phone | 200 → 208 ms | 0 → 0 | 64 → 65 ms | 80 → 80 ms |
| Fisheries | desktop | 76 → 76 ms | 0.050 → 0.001 | 0 → 0 ms | 48 → 48 ms |
| Fisheries | phone | 184 → 188 ms | 0 → 0 | 62 → 66 ms | 64 → 64 ms |
| About (new) | desktop / phone | 108 / 152 ms | 0 / 0 | 0 / 64 ms | — |

First-load JavaScript: Ocean Map 160 → 161 kB, Bloom 141 → 142 kB, Fisheries 137 kB, Sources 122 kB, About 121 kB. The self-hosted glyphs (1.3 MB in the repository) are fetched one range at a time; a typical page reads only the 77 kB Latin range.

## 7. Scientific and regulatory safeguards verified

- No notice is marked verified; "Not verified" stays on every notice list, the status line and the masthead; "A missing notice does not mean an area is open or seafood is safe" stays visible wherever notices are counted.
- The About page's research text is unit tested: it states "matches, but does not beat", says CoastWatch does not run the research models and has not been peer reviewed or independently reviewed, and contains no "outperform", "validate" or similar claim.
- Every page, About included, passes the "safe/open/loss only in negated sentences" check.
- Model, measured, official and historical keep their reserved colours; the new chips use the same tokens.
- No forecast value, mask, coastline or dataset changed (web only).

## 8. Outstanding concerns

1. **No data download.** Researchers can see every value (table) and the ERDDAP source, but cannot download what is in view.
2. **Notices are still unverified.** A person must review them (`cwp review-official`) before any public release.
3. **No independent HAB scientist review** of the application, the species tiers or the wording.
4. **At 375 px the phone status line wraps to two lines** (46 px instead of 35).
5. **Screen-reader testing** was by accessibility tree and axe, not a human with VoiceOver or NVDA.
6. **The private preview** can be reviewed only by an account with access to the Vercel team.
