# Integration contract: "Since you last checked", remembered waters and kept briefings

**Parties:**
- **#41 session (Terminal 1):** owns My Coast, the briefing, season status and next steps.
- **Fishermen-first session:** owns the files listed below.

**Branch:** `research/fishermen-first` (PR #47), stacked on `feat/coastal-redesign`. **Approved by the project owner on 2026-10-13:** this ownership split, and a North Coast commercial Dungeness pilot with recreational shellfish harvesters as the secondary group.

## 1. The mount: one import and one line

Agreed with #41 on 2026-10-13: the fishermen-first session adds this in its own commit, on top of #41's pushed head, and deletes the prototype route in the same commit.

In `components/coast/MyCoast.tsx`, inside `header()`, directly below `<AnswerLine … />`:

```tsx
import { SinceLastCheck } from "@/components/coast/SinceLastCheck";
// …
<AnswerLine clauses={…} />
<SinceLastCheck />
```

- **No props.** It reads `place`, `for` and `species` from the page's URL (`useSearchParams`), so it follows #41's link rules: a link's parameters win, and nothing is written to the URL.
- **Mount it in Coastal mode only.** In Research mode it is optional; my recommendation is to leave it out until the pilot shows it helps.
- **The prototype route** `app/coast/changes/page.tsx` is deleted in the same commit. It exists only so the feature could be tested before the mount. The cases in `tests/e2e/since.spec.ts` move to `/coast?…`; only the URL changes.

## 2. Files owned by the fishermen-first session (#41 should not edit them; ask instead)

| File | What it is |
|---|---|
| `src/lib/briefingSnapshot.ts` | A pure snapshot of one place, activity and species from CoastWatch's own published data, with agency, sample, model and check dates kept apart |
| `src/app/api/briefing/route.ts` | `GET /api/briefing?place&for&species` → the snapshot. It makes **no third-party request**: it uses only `loadData()`, which shares the 5-minute fetch cache with the page |
| `src/lib/changes.ts` | A pure diff of two snapshots, in 13 kinds (§4) |
| `src/lib/savedWaters.ts` | On-device memory (`cw-waters-v1`, at most 6 waters; a session baseline in `cw-changes-session-v1`). Every access is in try/catch |
| `src/lib/keptBriefing.ts` | Kept copies (`cw-kept-v1`, at most 5), the printable sections, and a self-contained HTML file |
| `src/components/coast/SinceLastCheck.tsx` | The block mounted under the answer line |
| `src/components/coast/KeptView.tsx`, `src/app/coast/kept/page.tsx` | `/coast/kept`, a print and read view. It deliberately has no app shell, so a saved copy is never shown beside live status |
| `tests/unit/changes.test.ts`, `tests/e2e/since.spec.ts` | 19 unit and 9 browser tests |

## 3. What my code depends on in #41's code

If any of these changes, `tests/unit/changes.test.ts` fails. Please tell me rather than editing my files.
- `resolvePlace`, `parseAudience`, `coastHref`, `ageDays` and `AUDIENCES` (`lib/coast.ts`);
- `associationsAt` and `AUDIENCE` (`lib/applicability.ts`), and `disputes` (`lib/official.ts`);
- `speciesById`, `recordNamesSpecies` and `statementTopic` (`content/species.ts`);
- `loadData`, and `StationLite.latest[]` with `variable` and `date` (`lib/data.ts`; your additive `recent` field is fine);
- `Mark` (`components/ui/Mark.tsx`), `formatDate` and `formatDateTimePT` (`lib/time.ts`), and `AGENCY_LINKS` (`content/coast.ts`).

## 4. Behaviour

| Situation | What the block shows |
|---|---|
| First visit to these waters (place, activity and species) | "Remembered on this device…", the time CoastWatch last checked the agencies' pages, "Keep a dated copy", and "Forget on this device". **No change list and no claims.** |
| Return visit with differences | "Since you last checked · <time>", then up to 4 items for the chosen species. "N more" reveals the rest (other species, data and model). |
| Return visit with no difference | "No change in CoastWatch's information for these waters since then. That does not mean nothing changed at the agencies." |
| Storage blocked or cleared | Nothing at all: the page is exactly as without the block |
| `/api/briefing` fails | One quiet line: "Could not compare with your last visit just now." |
| The page opened with no activity or species, for a place with remembered choices | A ≥44 px link: "Continue with Commercial fishing · Dungeness crab" (a link to `/coast?…`, never an automatic redirect) |

**Change kinds and the dates each one uses:**

| Kind | When | Date shown |
|---|---|---|
| `agency-statement` | The wording of an agency status statement changed (for example the Dungeness season status) | "CoastWatch checked" time; both wordings are quoted |
| `agency-issued` | A notice not listed before, **with an agency date on or after the last visit** | The agency's published or effective date |
| `agency-lifted` | Lifted, with a lift date on or after the last visit | The lift date |
| `release-pending` | A release appears to lift a notice here, not yet reviewed by a person | "CoastWatch checked" time |
| `listed-late` | Newly listed, or newly shown as lifted, by CoastWatch, but **dated by the agency before the last visit** (or undated) | The agency's date; the text says "Not new from the agency" |
| `removed` | No longer tied to these waters | none (the text says "does not know why") |
| `observation-new` | A newer sample at a station within 60 km | The **sample** date (and "published since" when it was sampled before the visit) |
| `forecast-new` | A newer C-HARM run | The issue date ("not a measurement of toxin") |
| `source-outage` / `source-recovered` | A source's latest update failed or recovered; an agency page could not be read | Attempt and success times |
| `became-stale` | The newest nearby toxin result passed 30 days; agency pages not checked for more than 2 days | Sample date or check time |
| `unavailable` | Official notices or a station were present last time and are missing now | none |
| `page-changed` | An agency page's content hash changed and nothing above explains it | "CoastWatch checked" time; "has not reviewed what changed" |

## 5. Kept briefing (I3)

"Keep a dated copy" stores the snapshot on the device. Then:
- **Save file:** a single HTML file with no scripts and no external requests. It opens without a connection (tested offline).
- **Print:** `/coast/kept?id=…`. Print hides the buttons and keeps the warning.

Each copy holds:
- the time it was saved (always with the year and time zone);
- the waters, activity and species;
- the agencies' statements and notices with their dates, plus pending releases;
- station sample dates, the model issue date, and failed sources;
- the hotlines;
- each source URL written out.

It always carries: *"This is a saved copy, not current information. Agency actions after it was saved may replace anything in it. It is not a legal authorization to fish or harvest, and not proof that seafood is safe."*

**Weather is deliberately not in the copy, only a link to NWS marine forecasts.** `api.weather.gov/robots.txt` disallows all paths, and the owner ruled on 2026-10-13 that a documented API is not automatically exempt (§7).

## 6. Recommendations: integrate or not

| Feature | Integrate? | Why |
|---|---|---|
| `SinceLastCheck` under the answer line (Coastal) | **Yes** | It removes the one task the app could not do (T5) and is quiet when there is nothing to say. Benchmarks: [07](07-benchmarks.md). |
| Remembered activity and species (the "Continue with…" link) | **Yes** | A returning user reaches their own species in 1 tap instead of 2–3 plus scrolling (T4). It never redirects by itself. |
| Kept briefing: Save file and Print | **Yes, as one secondary link** | Little cost and honest about age. Its value is a hypothesis, so measure it in the pilot (03 §8) before giving it more space. |
| `/coast/changes` prototype route | **No** | Test scaffold only; delete when mounted |
| Push, e-mail or SMS alerts | **No** | Not built and not claimed. Demand is unvalidated (03 §3, T-changed). |
| Weather in kept copies, or new NOAA calls from these features | **No** | Unresolved robots.txt question for `api.weather.gov` (§7) |
| Pre-season crab testing progress | **Not yet** | Needs a reviewed transcription pipeline for the CDPH PDFs (04 §2.1) |
| Disaster-assistance links | **Not now** | Only verified official programs, and only during a declared closure, without implying eligibility. No such closure is listed for these fisheries today (05 §8). |

## 7. Compliance notes for both sessions

- **These features call no third-party service.** `/api/briefing` reads only CoastWatch's own published artifacts.
- **`/api/conditions` (#41) calls `api.weather.gov`, whose robots.txt is `Disallow: /` for every agent.** It was fetched once on 2026-10-13. NWS documents the API for application use, but by the owner's rule that does not settle it. **Decision needed from the owner before production use** (#41 already lists "live NOAA conditions" as awaiting approval).
  - My walkthroughs now replay the recorded responses (`CW_CONDITIONS_FIXTURE`) and make no live NWS calls.
  - Earlier on 2026-10-13, before the ruling, my baseline walkthroughs triggered a handful of live calls through that route.
- **Other providers checked:**
  - NDBC's robots.txt blocks only named bots.
  - CO-OPS's API host has no robots.txt.
  - CDFW has none.
  - CDPH disallows only `/_layouts/`, `/_vti_bin/` and `/_catalogs/`.

## 8. Change protocol

- I won't edit #41's files.
- #41 won't edit the files in §2.
- Interface changes go by message to the other session.
- After #41 pushes, I re-run the six walkthroughs against its head, with this block mounted by the line in §1 in a local, unpushed build, and report the results ([07](07-benchmarks.md)).
