# Neritic v12 research C: information architecture and usability

Audit of `feat/neritic-v11` at `c02f460cb`, run against `localhost:3410` (recorded fixtures, `NERITIC_PILOT=1`) in Chrome through Playwright, 1440x900 and 390x844, fresh contexts with empty storage. Source paths are relative to `coastwatch-web/src/`. Evidence is in `/private/tmp/claude-502/-Users-yashnilmohanty-Desktop-habs-forecast/597c2abb-016b-4c09-bb3a-b1e2d415299e/scratchpad/audit-C/` (`census.json`, `walk.json`, `measure.json`, PNGs).

**Limits of this evidence.**
- The fixtures are 33 to 42 days old, so "(old)" and Explore's "no current forecast" state are heavier here than on live data. "Test data" in the shell is fixture-only.
- Counts are DOM text inside the viewport after settling, ±10 words (some assistive text is counted). `docs/coastwatch/v11/ia-audit.md` counted from screenshots, so comparisons with it give direction only.
- Latency is localhost. Slow-network flashes are inferred from code.
- No real fisherman or researcher took part. The three walks are an expert simulation.

**Verdict.** v11 reduced density but left the structure alone, and the structure is the complaint. One person with one port is sent through two pickers, two port stores, two "briefings" and three offline features under four names. What opens first depends on the URL and a build flag. Navigation names kinds of page rather than things a person wants to do.

## 1. Route and entry map

| Route | What it is | Ways in | Redirects and forks |
|---|---|---|---|
| `/` bare | Front door | Typed URL, brand logo on every page (`TopBar.tsx:224`), kept-copy header (`KeptView.tsx:44`) | **Pilot:** 307 to `/today` (`app/page.tsx:20`, `pilot/frontDoor.ts:73`). **No flag:** renders Explore with Splash. Campaign tags alone (`utm_*`, `fbclid`, `ref`, `source`...) still redirect |
| `/?view=map`, `/?<any param>` | Explore | Explore tab (pilot), Today bloom link, My Coast map links (`?sel=&from=`), About, shared links | Renders. `view=map` is pilot-only scaffolding. URL rewritten with `layer`, `at`, `sel`, `fp` via replaceState (`Explore.tsx:311–323`) |
| `/today` | Port picker "Where do you fish?" | Redirect from `/`, Today tab with no saved port, `/today?pick=1` (H1 chevron) | Server renders the picker, then a **client** `router.replace` to the last or home port (`PortPicker.tsx:27–33`). Measured URL change at 367 ms |
| `/today?place=port-N` | Today at one port | Picker, directory, Today tab (`readTodayHref`), Explore "Open today's briefing", `/account`, Ask | `?f=` overrides fisheries. Any California port; alerts and Ask are 4 pilot ports only |
| `/coast` | My Coast landing "Where are you fishing?" | My Coast tab with no home, **PWA `start_url`** (`app/manifest.ts`) | **Client** replace to home briefing if `cw-home-port` is set (`Briefing.tsx:124`). `?mode=research` goes to `/research?place=` (`app/coast/page.tsx:15`) |
| `/coast?place=…` | Trip briefing | My Coast tab with home, Landing, Explore "Plan a trip"/"Brief this"/"Open the briefing", species search (hard `location.assign`, `Explore.tsx:682`), offline page | Place may be `port-N`, a harvest site or `point-lat,lon`. `?sheet=trip` opens the worksheet view; `?focus=`, `?date=`, `?species=`, `?for=`, `?home=` |
| `/coast/kept`, `?id=` | Dated copies | My Coast, SinceLastCheck | Own bare header; not in the shell |
| `/research` | Station record; default Santa Cruz Wharf | Research tab, "History in Research", "Its record in Research", More > **"Bloom stations"** | `?station ?place ?variable ?day ?start ?end ?range ?from`, tabs by hash (`Research.tsx:152–176, 349`) |
| `/ask?place=` | Ask page | More menu, footer, Today action on non-pilot ports (sheet on pilot ports) | Defaults to first pilot port |
| `/account?place=` | Ports, alerts, inbox, trip log | More menu, bell, Today "Set an alert" | none |
| `/fisheries` | Status by species + landings history | More menu | Legacy `AppShell`; content varies by `useMode` |
| `/bloom` | v2 station page | **No link from `v4/` or `pilot/`**; only `components/` and `content/copy.ts` | Orphan duplicate of Research, legacy shell, 452 words on first screen |
| `/sources`, `/about` | Data status; about | More menu, footer, trust pill | `/about` hosts PurposeSwitch |
| `/privacy /terms /sms-terms` | Legal | Footer | none |
| `/api/{ask,briefing,conditions,evidence,sms,alerts/fanout}` | Services | Client, cron, Twilio | Keep paths |

### Openings and first-run states

| # | State | Code | Trigger and dismissal | Measured |
|---|---|---|---|---|
| A | Splash film "Sounding" | `v4/shell/Splash.tsx`, `SplashGate` in `MotionRoot.tsx`, theme.css | `pathname === "/"`, first visit, no `nx-/cw-/neritic-` keys. `nx-splash-seen-v2` is set on the first visit to **any** page. Off: `nx-splash-off-v1`; test: `nx-splash-force-v1`; suppressed for `navigator.webdriver` | Bare map ground, mark at 2.2 s, lockup at 3.5 s, map by 5.0 s |
| B | Picker hero | `pilot/today/PortPicker.tsx` + OceanStage `arrival="coast"` | Every `/today` without a port | Text at 0.4 s; map fades in by 3.5 s |
| C | Port arrival | `OceanStage` `arrival=place_id`, three `data-beat` reveals | Per port, per session (`nx-stage-arrived:*`) | Port page text at 0.4 s, map by 3.5 s |
| D | Picker descent | `PortPicker.tsx:37–44` (`journeyTo`) | Click on a pilot port only; directory links skip it | 1,300 ms before navigation |
| E | My Coast landing | `v4/coast/Landing.tsx` | `/coast` without place or home | Static SVG coast; **a different map from B** |
| F | Explore coach mark | `Explore.tsx:940–985`, `nx-explore-coach-v1` | Data layer at state scale; absent on the seafloor fallback | The only "how to use this" text in the app |
| G | Purpose switch | `components/welcome/PurposeSwitch.tsx`, `lib/purpose.ts` | About page only | Still offers "a short guided tour" that no longer exists |
| H | Dead code | `components/entry/Welcome.tsx`, `components/map/Tour.tsx` | **Never mounted** (grep: no `<Welcome`, no `<Tour`). Only `TOUR_SEEN_KEY` is still imported | Delete |
| I | Install hint | `OfflinePanel.tsx:305`, `nx-a2hs-hint-v1` | Offline panel | n/a |

**Where landing experiences overlap, precisely.**
1. **A vs B.** Under the pilot flag, A cannot appear at `/`. It plays only when a stranger's *first ever* page is a map deep link, so the people who get the film are those holding a shared link, who wanted the content (`census-open-map-d-t2200.png` vs `census-open-root-d-t1200.png`). Without the flag the same URL gives A then Explore and never B. The product has two front doors, chosen by env var and query string.
2. **B vs E.** "Where do you fish?" and "Where are you fishing?" are the same screen built twice: different port lists (4 + directory vs 6 + search), different maps, different stores.
3. **A, B and C are three cinematic languages**: drawn chart to logo, map fade behind a hero, map plus staggered text. D adds a fourth, the camera flight, which only one kind of link gets.
4. **PWA.** An installed app opens at `/coast`, the web at `/today`, a shared link at Explore.

**Transitions are inconsistent because six mechanisms coexist:** the Splash handoff; View Transitions for path changes only (`v4/motion/routes.tsx`, `routeFilter.ts:13`, so a port switch on `/today` is a hard cut); OceanStage beats; the camera flight; `cw-fade-in` on My Coast; legacy `PageIn` on `/bloom`, `/fisheries`, `/sources`. Tab glide state is yet another key (`nx-nav-pill-v1`).

## 2. The four destinations

**Stranger test.** Could someone say what it is for from the first screen, without the tab label?

| | Currently for | Overlaps | Stranger test |
|---|---|---|---|
| **Today** | One port now: status, sea, toxin tests, bloom, keep/alert/ask. Good lede | My Coast: sea, status, what changed, offline, tide. Fisheries page: status by species | **Mostly yes** on the port page; the picker sentence is good. "Today" does not tell you it is port-scoped |
| **Explore** | Layer map, place inspector, time rail, field packs, area tool | Today (place summary), My Coast (place inspector, map), Research (history, cell reads, transects) | **No.** The H1 is screen-reader only. On fixtures, the first screen is seafloor plus a warning. "Explore" names an activity with no object |
| **My Coast** | Landing + one-trip briefing: status for one species, wind/seas/tide, 60-h lanes against limits, checklist, economics, evidence, dock value | Today, almost entirely, for the status and sea. "My" promises personal, but nothing personal exists until a hidden "Your home port" action | **No.** Landing says "one trip" (good) but is indistinguishable from Today's picker. Briefing H1 is the same "Moss Landing" as Today |
| **Research** | Station record vs C-HARM, provenance, native grid, exports | `/bloom` (duplicate), Explore inspector | **Yes for a scientist, no for anyone else**, which is acceptable. It opens on Santa Cruz Wharf whatever port you came from |

**Per screen: what am I looking at, why does it matter to me, what can I do next.**

| Screen | Looking at | Why it matters | What next |
|---|---|---|---|
| Picker `/today` | Clear | Clear ("official status for your fisheries...") | Clear, 4 ports + browse. Non-pilot fishers must discover "Browse". No word about who runs this or that it is free |
| Today port | Clear: port, top fact | Clear for the lede; four glance cells need decoding ("5.5 of 30 ppm", "4 flagged") | **Weak.** Keep/alert/ask at y=2273 (desktop), 2935 (phone). No path from "seas 4–6" to a trip other than "5 days against your limits" at the end of the Sea section |
| Explore default | **Unclear** | **Absent** | Unclear: two buttons about stale data, tiny numbered clusters. Coach mark only when a data layer is on |
| Explore, place open | Clear: three statements | Partial | **Confusing:** "Open today's briefing" (goes to Today), "Plan a trip" (My Coast), "History in Research". For a water point the same primary is "Open the briefing" and goes to My Coast |
| My Coast landing | Partial, looks like Today's picker | Good sentence | Clear |
| My Coast briefing | Clear on phone; dense on desktop (367 words, 67 numbers) | Clear, and "Nothing here says a trip is allowed" is honest | Weak: limits, checklist, economics and keep are all below the fold. "Save for offline use" is in the page footer |
| Research | Clear to a specialist | Unstated | Good: tabs, range, Table, Export, Lineage |
| Ask | Clear | Clear | Clear, but 210 words on the first screen, with a disclaimer paragraph above the question field |
| Account | Clear | Clear | Three "Not yet connected" states and legal consent above the fisheries or ports a person came to set |
| Fisheries | Two H1s, two purposes (status and landings history) | Status repeats Today by a different route | Unclear why it is separate |

**Walk notes.**
- *First-time fisherman.* `/` to picker to Moss Landing works in one tap and is the best moment in the product. He then taps My Coast and is asked where he fishes (`walk-j1-*-4-mycoast.png`). Explore then shows no port, Research shows Santa Cruz Wharf.
- *Returning fisherman.* `/` lands on his port in about 370 ms after a picker render. There is deliberately no go/no-go, which is right. But the decision needs status (Today, first screen), the sea against his limits (My Coast lanes, 60 h, limit inputs), and tide or buoy detail (either). Today's Sea starts at y=936 (phone 1275); the limits-based view is a tab away and repeats the port header and status.
- *Researcher.* Explore to tap water to "History in Research" carries station, variable, day and back-link, and browser Back restores the selection exactly. This is the best-engineered flow in the app. The defect: a tap about 340 km offshore landed on Bodega Marine Lab buoy with no distance shown (`walk-j3-d-3-research.png`), and the requested pDA silently became *P.-n. seriata*.

## 3. Capability inventory (must survive consolidation)

Target column uses the structure in section 6.

| Capability | Where now | Target |
|---|---|---|
| Official status per fishery, sheet with cause/area/review/source | `pilot/today/Status.tsx`, `pilot/facts.ts` | Port |
| Edit followed fisheries (`?f=`, picker) | `TodayView` FisheryPicker, `prefs.ts` | Port; one store |
| Whale rules / RAMP sheet | `Status.tsx` WhaleSheet | Port |
| What changed since last check | `pilot/today/Changed.tsx`, `components/coast/SinceLastCheck.tsx` | Port; one implementation |
| Toxin test charts, per-site, CDPH link | `pilot/today/Tests.tsx`, `Charts.tsx` | Port |
| C-HARM + outlook summary with "not toxin" wording | `pilot/today/Bloom.tsx` | Port |
| Sea now: NWS, buoy, tide, advisory | `pilot/today/Sea.tsx`, `marine.ts`, `/api/conditions` | Port |
| 60-h lanes, personal wind/sea limits | `v4/coast/Lanes.tsx`, `TideStrip.tsx`, `lib/trip.ts` (`cw-sea-limits-v1`), `/api/briefing` | Port, replaces Today's strip |
| Trip inputs: activity, species, day | `v4/coast/Briefing.tsx:361–363`, `?for ?species ?date` | Port trip control |
| Non-port places (harvest sites, water points) | `lib/coast.ts`, `resolvePlace` | Place page, status section degrades. **Today's model is `port-*` only** (`app/coast/page.tsx:20`) |
| Before-you-go checklist, per trip | `Briefing.tsx:199–216, 418` | Port |
| Trip economics, scenarios, scenario link, compare | `v4/coast/economics/*`, `?sheet=trip` | Port drill-in sheet |
| Season list, CDFW list, dock value history | `Briefing.tsx` folds, `v4/coast/season.ts` | Port folds |
| Evidence claims, unknowns ("Not known here") | `v4/coast/Matters.tsx`, `Briefing.tsx:496` | Port fold |
| Notice geography on map; harbour map with limit lines | `v4/coast/HarbourMap.tsx`, OceanStage `lines` | Port map |
| Set home port | `Briefing.tsx:750–823`, `v4/coast/home.ts` | Port header, explicit |
| Change port, directory of all CA ports | `PortPicker`, `PortDirectory.tsx`, `?pick=1` | Shell-level port switcher |
| Save offline: port page | `pilot/today/offline.ts`, `neritic-pack-today` | One "Keep offline" |
| Offline field packs, PMTiles, quota, resume, delete | `v4/offline/*`, `public/sw.js`, `public/offline/monterey` | Same action, plus Map |
| Dated kept copies, print, `?id=` | `lib/keptBriefing.ts`, `components/coast/KeptView.tsx`, `/coast/kept` | Keep; a named option of Keep offline. Keep the SW offline routes in step with any URL change |
| Copy link to trip | `Briefing.tsx:692` | Port |
| Alerts: bell, inbox, triggers, device check | `pilot/alerts/*`, `AlertCheck`, `/api/alerts/fanout` | Shell + Account |
| Web push, SMS opt-in + consent | `PushPanel.tsx`, `AccountApp.tsx:289`, `/api/sms`, `/sms-terms` | Account; legal text unchanged |
| Account: role, vessel, sync, trip log, feedback | `pilot/account/*`, `TripLogForm`, `Feedback.tsx` | Account |
| Ask: chat, sheet, citations to fact anchors, SMS preview, evals | `pilot/ask/*`, `/ask`, `/api/ask` | Sheet from Port; `/ask` stays |
| Map layers: C-HARM (3 variables, leads), satellite (multi, OLCI, VIIRS, age, sensor), outlook, SST, HF currents, seafloor | `v4/explore/LayerPanel.tsx`, `layers.ts`, `FieldDock.tsx`, `FlowControls.tsx` | Map |
| Time rail, play, day ticks | `v4/explore/Timebar.tsx` | Map |
| Opening-layer logic (current C-HARM, else satellite, else honest empty) | `lib/opening.ts` | Map |
| Inspector: harbour, station, point, footprint (`fp`), evidence lines | `v4/explore/Inspect.tsx` | Map |
| Search: harbours, stations, regions, species, sites | `v4/explore/Search.tsx` | Shell or Map |
| Area select, share, field pack tools | `Explore.tsx:1545–1605`, `AreaDrag.ts` | Map |
| Station record, linked model panel, range, crosshair | `v4/research/Research.tsx`, `v4/charts/LinkedSeries.tsx` | Research |
| Native grid: cells, transects, area stats | `Research.tsx:685–905`, `ObsTools.tsx`, `transect.ts`, `obsGrid.ts` | Research |
| Exports CSV with provenance (window, model, cell, transect, area) | `v4/research/exports.ts`, `parts.tsx:61` | Research |
| Provenance, QC, lineage, citation, table view | `Research.tsx:629, 906`, `Lineage.tsx`, `layerLineage.ts` | Research |
| Fishery landings history and exposure | `components/fisheries/FisheriesExposure.tsx`, `?unit ?tiers ?group ?year` | Research section; status half folds into Port |
| Notices registry popover, disputed list, status centre | `v4/shell/Notices.tsx` | Shell |
| Data status, source list, agency phone numbers | `/sources` | Keep |
| Daylight mode | `v4/shell/daylight.ts`, TopBar (phones get a sun button) | Shell |
| Honest readiness labels | `pilot/readiness.ts`, `ReadinessBadge.tsx` | Keep |
| Keyboard | `/` search (`Search.tsx:122`), arrows/Enter/Esc in results, Esc closes panel, `a` area tool with map focused (`Explore.tsx:668`), rail arrows/Home/End/Space (`Timebar.tsx:498–502`), layer radio arrows (`LayerPanel.tsx:162`), skip links, sheet Esc | All kept; document them |
| Reduced motion, poster fallback, no-WebGL note | `v4/motion/reduced.ts`, `v4/stage/fallback.ts`, `StagePoster.tsx` | Keep |
| Legal/honesty copy | "Nothing here says a trip is allowed", "not toxin", "not verified", "a gap, not a low value", independence line | Non-negotiable |

**Deep links that must keep resolving:** Explore `region port var lead layer inspect sel at fp from chl age sensor`; Research `station place variable day start end range from #tab`; Coast `place for species date sheet focus home mode`; Today `place f pick` and fact hashes from Ask; `/ask?place`, `/account?place`, `/bloom?station var range year`, `/fisheries?unit tiers group year`, `/coast/kept?id`; `?mode=` on legacy pages.

## 4. Distraction census: first screen

`Words / numbers / links / controls`. Trust = occurrences of "not verified" and old-age wording in the first screen. Shell alone is about 20 words on every page ("Test data · Research preview · 11 notices in CA · Not verified · Some data is old" + tabs).

| Page | 1440 | 390 | Not verified | Old | Worst repeats |
|---|---|---|---|---|---|
| Picker `/today` | 93/3/10/4 | 99/5/10/5 | 1 | 0 | None. Best screen |
| Today port | 227/38/11/11 | 161/27/11/10 | 4 | 9 | "Check is old" x3 in view. Page-wide: "(old)/is old" x12, "not toxin" x7, "seas 4–6 ft" x6, the delay sentence x3 |
| Explore, seafloor fallback | 181/35/8/11 | 114/11/6/10 | 1 | 4 | The stale message and two stale buttons |
| Explore, chlorophyll | 199/32/8/11 | 107/8/6/10 | 1 | yes | Coach mark + stale notice + legend together |
| Explore, harbour open | 314/57/11/21 | 123/14/9/12 | 2 | 4 | "Tue Nov 10" x3, "Details" x5 |
| My Coast landing | 94/4/18/4 | 80/1/17/5 | 1 | 0 | Port names on map and in list |
| My Coast briefing | 367/67/10/9 | 217/17/10/9 | 3 | 2 | "6a noon 6p" x3; status repeats the crab line |
| Research | 296/66/6/18 | 171/33/6/18 | 1 | 2 | "Measured at the pier" x3 |
| Ask | 210/1/7/13 | 157/1/7/8 | 1 | 0 | Moss Landing x4 |
| Account | 278/3/9/6 | 196/3/8/6 | 1 | 0 | "Not yet connected" x4 |
| Fisheries | 276/13/14/16 | 167/1/8/8 | 2 | 0 | "Dungeness crab: closed list CDFW" x4 |
| `/bloom` | 452/42/7/18 | 207/17/7/6 | 2 | 0 | "pDA not measured here" x7 |
| Sources | 198/7/12/3 | 180/1/10/4 | 1 | 0 | none |
| About | 207/2/7/5 | 166/2/7/5 | 1 | 0 | none |

Against v11's audit (method differs): Today ~58 to 38 numbers and My Coast landing ~20 to 4 improved. **My Coast briefing (~40 to 67) and Explore with a place (~40 to 57) did not, and Research is unchanged.** Density work was uneven because each page was tuned alone.

The phone shell clips its own status to "So…" (`census-today-port-m.png`). The word "Not verified" is honest and must stay; what repeats around it is the problem.

## 5. State continuity

| State | URL | localStorage | Crossing pages (measured) |
|---|---|---|---|
| Port on Today | `?place` | `nx-today-place-v1` (implicit, last visited) | My Coast tab ignores it: `/coast` asks again |
| Home on My Coast | `?place&for&species` | `cw-home-port` (explicit, with trip) | Today uses it only if there is no last place. Viewing Monterey's briefing, the Today tab still points at Moss Landing |
| Home in Account | none | `nrt-pilot-account-v1.home_place_id`, synced | **Does not feed Today or My Coast** |
| Fisheries | `?f` | `nx-today-fisheries-v1`; separately account fisheries; separately species in `cw-home-port` | Three lists for one fact |
| Explore selection, viewport | `sel fp at` | none | Lost via tabs: Explore tab is a fixed `/?view=map`; returning gave `/?layer=none&view=map`. Browser Back restores it exactly |
| Explore layer, time | `layer var lead` | `nx-currents-view-v1` (view only) | Lost via tabs; not offered to Research |
| Research station, range | `station start end range day` | none | Tab returns to Santa Cruz Wharf. `/research?place=port-592` would have opened Monterey Wharf; the tab never passes it |
| Trip day, limits, checklist | `date` | `cw-sea-limits-v1`, per-trip tick keys, `cw-waters-v1`, `cw-kept-v1` | Today knows none of them |
| Device flags | none | `nx-splash-seen-v2`, `nx-explore-coach-v1`, `nx-daylight-v1`, `nx-ask-seen-v1:*`, `nx-today-seen-v1:*`, `cw-mode-v1`, `cw-purpose-v1`, `cw-intro-dismissed-v1` (dead), `cw-camera-tier-v1`, `nx-a2hs-hint-v1` | `cw-` / `nx-` / `nrt-` prefixes reflect three eras |
| Session | none | sessionStorage `nx-stage-arrived:*`, `nx-nav-pill-v1`, `cw-glide-*` | Motion only |

**Rule implied:** contextual links carry state well (`crossHrefs`, `Inspect.tsx:1190`). Global tabs throw it away. The fix is to make tabs behave like contextual links.

## 6. Target IA

### Four destinations or fewer?

**For keeping four.**
- "Now at my port" and "planning one trip" are distinct questions.
- Separate pages stay shorter: merging Today (2,793 px) and My Coast (3,443 px) needs discipline.
- Existing tests and deep links assume it; `ia-audit.md` §3 already drew an ownership line.
- The scope difference is real: Today is curated ports, My Coast accepts any place.

**Against.**
- It is the same person, port and facts. About four of Today's seven sections recur in My Coast.
- A go decision needs both.
- Two heroes and two stores exist only to maintain the split.
- v11 tried to clarify by ownership and the owner still cannot tell what a user should do. The labels cannot be fixed by wording because the two pages do not differ by intent, only by time horizon. A time horizon is a control, not a destination.
- "My Coast" also claims the personal territory that Today actually delivers.

**Recommendation: three destinations, plus a shell-level port identity.**

| Tab | Intent it expresses | Replaces | First screen answers |
|---|---|---|---|
| **Your port** (shows the port's name once chosen: "Moss Landing") | "What's going on at my port, and should I look harder at going out?" | Today + My Coast | Top official fact, sea today and tonight, what changed |
| **Map** | "Where is it happening?" | Explore | The field, with one dock |
| **Research** | "Show me the record" | Research + `/bloom` + landings history | Station record |

- Ask is a persistent action on Port; the alerts bell stays in the shell. Account, Sources and About stay in More.
- On the Port page, a **day-and-trip control** replaces the page boundary: default "Today"; choosing a day, species or limits fills in the lanes and checklist in place. Economics stays a drill-in sheet (`?sheet=trip` exists). No go/no-go verdict, ever; "Nothing here says a trip is allowed" stays on every width.
- The section order on Port follows the v11 evidence ladder: changed, official status, sea and lanes, tests, bloom, keep/alert. Keep/alert moves up: a compact action row under the header, not y=2273.
- **Fisheries status** folds into Port. **Fisheries landings** moves to Research.
- **Offline:** one verb, "Keep offline", with one dialog offering *this page* (today's pack), *this area with map* (field pack, also from Map) and *a dated copy to print*. The engines remain separate.

### One canonical entry

| Visitor | Path |
|---|---|
| **First visit at `/`** | One page, the current picker (best screen in the product), with one added line on who this is for, that it is free, and that agencies come first. Offer "Open the map" and "I'm a researcher" as quiet links. Pick a port, then Port. The flight (D) is the only arrival animation, and it applies to every port link including the directory |
| **Returning at `/`** | Directly to their port. Use an inline pre-paint script like `SplashDecide` that calls `location.replace` before the picker paints. **Do not use a cookie for this:** the app promises device-only storage ("Nothing is sent anywhere", account copy and `prefs.ts`). A cookie sends the port to the server and would break that promise |
| **Deep link** | Straight to that content. **Never a film.** Shared links exist to deliver content |
| **Installed PWA** | `start_url: "/"`, same logic |

**The Sounding film:** remove it from all deep-link paths. The owner's complaint is that the opening and the page are separate. The options are to delete it (my recommendation; it costs 3.5–5 s on the people least patient for it) or to make it a sub-second brand beat resolving into the picker's own map. Whichever is chosen, there must be one opening, on one surface, with one transition language, and `NERITIC_PILOT` must stop forking the front door.

**State model.** One device record `nx-port-v1 { place, fisheries, for, species, limits }`, migrated on first read from `cw-home-port`, `nx-today-place-v1`, `nx-today-fisheries-v1`; account sync writes the same record. Tabs become contextual: Map opens at the current port, restoring last layer and view for the session; Research opens the nearest station with `place` and the Port's `from` for a Back. Picking a port anywhere sets the current port; "home" is that record, and an explicit pin is optional.

**Motion.** One route transition for path *and* port changes; drop `PageIn`, `cw-fade-in` and per-page beats in favour of it.

### Redirect plan (permanent unless noted)

| Old | New |
|---|---|
| `/` + any Explore param (`namesExplore` logic) | `/map` + same params |
| `/?view=map` | `/map` (drop `view`) |
| `/today` | `/` (front-door logic) |
| `/today?pick=1` | `/?pick=1` |
| `/today?place=X&f=Y` | `/port/X?f=Y` |
| `/coast` | `/` |
| `/coast?place=X&for&species&date&sheet&focus` | `/port/X` + same params |
| `/coast?mode=research&place=X` | `/research?place=X` |
| `/coast/kept[?id]` | keep the path (service worker, saved copies) or update `KEPT_PAGE` and `sw.js` in the same change |
| `/bloom?station&var&range&year` | `/research?station&variable&range` (map `var` to `variable`; `year` to `start/end`) |
| `/fisheries?unit…` | `/research/landings?unit…`; the species status view to `/port/X` or a section anchor |
| `/ask`, `/account`, `/sources`, `/about`, legal | unchanged |

`X` keeps today's keys (`port-592`, `point-lat,lon`, sites), so `/port/` must accept non-port places or be named `/place/`. Saved offline copies are keyed by URL: **the service worker's page cache must be migrated, or old packs will stop opening** (`public/sw.js`, `v4/offline/packs.ts`, `pilot/today/offline.ts`). Tests encoding the old front door: `pilot-front-door.spec.ts`, `splash.spec.ts`, `today-picker.spec.ts`, plus the table in `ia-audit.md` §8.

## 7. Ten most damaging problems

Screenshots are under `/private/tmp/claude-502/-Users-yashnilmohanty-Desktop-habs-forecast/597c2abb-016b-4c09-bb3a-b1e2d415299e/scratchpad/audit-C/`.

1. **Two front doors, decided by a query string and an env var.** A first-time visitor at `/` gets a picker; one on a shared map link gets a ~4 s film and then a map; an installed app gets My Coast. *Evidence:* `census-open-map-d-t2200.png`, `census-root-d.png`. *Fix:* one entry (section 6); film off deep links; remove the `NERITIC_PILOT` fork; PWA `start_url: "/"`.
2. **Today and My Coast split one decision across two pages.** Status and short sea on one, limits and lanes on the other, each with the same port hero. *Evidence:* `walk-j2-d-2-scroll.png`, `walk-j2-d-3-mycoast.png`. *Fix:* merge into Port with a day-and-trip control.
3. **The app asks where you fish twice.** Immediately after picking Moss Landing, the My Coast tab asks again with a different picker. *Evidence:* `walk-j1-d-4-mycoast.png`, `walk-j1-m-4-mycoast.png`. *Fix:* one port record, one picker.
4. **Navigation and link names don't tell you what happens.** "Briefing" means Today in "Open today's briefing", and My Coast in "Open the briefing" and "Brief this". "Plan a trip" is also My Coast. "Bloom stations" opens Research. Offline is "Keep it offline", "Field pack", "Save for offline use" and "Keep a dated copy". *Evidence:* `walk-j3-d-2-tap-water.png`, `phone3-1-harbor.png`. *Fix:* tabs by intent (Your port, Map, Research); one verb per capability; a link names its destination.
5. **Tabs discard context and can silently show a different port.** On Monterey's briefing, Today opens Moss Landing; Research always opens Santa Cruz Wharf; returning to Explore loses selection, viewport and layer. *Evidence:* `walk-cont-d-explore-after-research.png`, `walk-j1-d-6-research.png`. *Fix:* contextual tab hrefs; one current-port state; session restore for Map.
6. **The actions a fisherman returns for are at the bottom.** Keep offline, alert and Ask sit at y=2273 (phone 2935), after everything else; My Coast puts "Save for offline use" in its footer. *Evidence:* `j1-d-3b-port-full.png`, `j1-m-3b-port-full.png`. *Fix:* a compact action row under the port header; a persistent Ask.
7. **The trust cluster and repeated caveats drown the one fact.** Five status items in the bar before content, clipped to "So…" on a phone; on Today, "(old)/is old" x12 and "not toxin" x7. Each caveat is justified alone; together they read as noise and cost trust. *Evidence:* `census-today-port-d.png`, `census-today-port-m.png`. *Fix:* one bar pill keeping "Not verified" in words; caveats once per group and beside the claim (the `ia-audit.md` §3 conditions); stale called out only by exception.
8. **Explore never says what it is or what to do.** The heading is screen-reader only; the fallback state shows seafloor and a warning with no purpose statement; the coach mark doesn't appear there at all. Part of this is fixture staleness, but the honest-empty state is real on any bad data day. *Evidence:* `census-explore-d.png`, `ex-chl-d.png`. *Fix:* a visible one-line title and the field dock; the coach mark on every first map visit; empty state offers the newest dated layer as the primary action.
9. **Phone layout collisions on the two most-used pages.** On Today the freshness line runs 10 px into the port title (measured: freshness bottom 375, H1 top 365). My Coast's phone briefing opens with a map band, so the title and status start well down the screen. *Evidence:* `census-today-port-m.png`, `census-coast-brief-m.png`. *Fix:* freshness by exception, on its own line below the lede; on phones, words first and the map second.
10. **Cross-links to Research can mislead.** A tap about 340 km offshore opens a Bodega buoy record with no distance or "nearest" statement, and requested pDA silently becomes another variable. Harbours do get "Arrived from Moss Landing · nearest station Monterey Wharf, 24 km", so the pattern exists. *Evidence:* `walk-j3-d-3-research.png`, `walk-j3-m-3-research.png`. *Fix:* apply that "arrived from" line to points with the distance; state when the asked variable isn't measured there; past a stated distance, offer "no nearby station" rather than a record.

**Also worth fixing.** `/bloom` is an orphaned, inconsistent duplicate. `/account` puts consent before setup. Ask puts a disclaimer paragraph above its question field. `Welcome.tsx`, `Tour.tsx` and the "guided tour" wording are dead.

**Credit where due, keep these:** the picker's sentence; the Today lede; Explore's Back behaviour and `crossHrefs`; Research's linked panels; "Nothing here says a trip is allowed"; and the honest readiness labels.
