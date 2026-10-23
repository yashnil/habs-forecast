# Neritic v8: market and design research (2026-10-22)

Three research passes, run in parallel on 2026-10-22, with web sources:
1. **HAB, water-quality, ocean-portal and fishing apps**: the market this product competes in.
2. **What makes weather, marine and geophysical apps beautiful**: the craft bar.
3. **Frontend design tooling**: impeccable, the Vercel Web Interface Guidelines, Taste Skill, 21st.dev, Magic UI, shadcn/ui, Spline and others.

Claims a pass could not verify against a live source are marked *unverified*. Prices come mostly from third-party comparisons (2024–2025) and should be checked before anyone quotes them.

## 1. The market

### HAB and water quality

| Product | Core job | Patterns worth taking |
|---|---|---|
| **C-HARM v4** (SCCOOS/CeNCOOS) | Daily probability maps 1–3 days ahead for *Pseudo-nitzschia* > 10,000 cells/L, pDA > 500 ng/L and cDA > 10 pg/cell | States the threshold behind every probability. Warns that toxin can occur below it. |
| **California HAB Bulletin** (monthly) | Model GIFs, pier samples, IFCB, CDPH results and strandings, as a narrative | Model and observation in separate sections, compared explicitly. "Not intended to inform … of the presence or lack of risk"; it points to CDPH for advisories. |
| **NOAA Gulf red tide forecast** | Beach-by-beach respiratory risk, every 3 h out to 30 h | An **impact-based legend**: each level is defined by who is affected. |
| **Lake Erie HAB forecast** | Animated loop from the satellite image date out to 96 h or more | The loop is anchored at the observation date. Black means no chlorophyll, grey means no data. "For guidance only". It says plainly that there is **no operational toxin forecast**. |
| **CDPH shellfish advisories / CDFW closures** | The authority: county advisories by press release; domoic closures; Dungeness openers by zone | This is the decision fishermen actually wait for. |
| **Heal the Bay NowCast** | Daily bacteria predictions for 20+ beaches | A poor prediction is labelled **advisory, not a closure**. |
| **Swim Guide** | Pass/fail pins with the last test date and the % of tests passed over the past year | Reviews: **stale data destroys trust** (3.6★). |
| **Florida FWC red tide** | Sample map for the last 8 days, weekly report | Legend bins list their **likely effects**. Data is called preliminary; satellite "complements, does not replace" sampling. |
| **bloomWatch / CyAN** | Citizen bloom reports (a 3-photo protocol), satellite cyanobacteria index | Reports are labelled "possible" and routed to agencies. |
| **EOMAP eoapp AQUA, BlueGreen BGi** | Commercial satellite water-quality platforms | Per-site **threshold email alerts**, history back 35–40 years. |

### Ocean portals and fishing intelligence

| Product | Pattern |
|---|---|
| **Copernicus MyOcean Pro** | Time and depth sliders drive both map and graphs. Click a point for a time series with CSV export. Particle currents. Climatology synced to the timeline date (v16). |
| **NASA Worldview** | Date slider, animation and GIF export. **A/B compare** by swipe, opacity or spy. |
| **Global Fishing Watch** | A timebar with a draggable range and presets. Hovering a track on the map highlights it in the timebar, and the reverse. |
| **ROFFS, Hilton's, SatFish, RipCharts, FishTrack** ($129–$2,268/yr) | SST and chlorophyll imagery, offline downloads, waypoints. **None integrates HAB, biotoxin or closure status.** RipCharts reviews complain about cloud-blocked imagery. |
| **Fish Rules** (free) | Regulations by location, offline, tied to the permit. Regulatory closures only, Gulf and South Atlantic. |
| **Navionics, Orca** | Charts and routing, animated multi-model weather, condition-change warnings. |

**The open niche:** no fishing-intelligence product joins health closures and HAB evidence with ocean conditions in one place. That is Neritic's position, and it is worth defending with craft.

### The capabilities that recur, ranked for a fisherman's HAB app

Each row was checked against Neritic's v7 branch (#77) before this pass.

| # | Capability | Kind | Neritic v7 | v8 |
|---|---|---|---|---|
| 1 | Official closure/advisory status by place, with source and date | table stakes | ✔ (notices, unverified, with links) | unchanged |
| 2 | Data age on everything; a stale state | table stakes | ✔ | the front door's readings say their age |
| 3 | Map with a tap readout (value, units, date, source) | table stakes | ✔ (inspect panel, hover readout) | unchanged |
| 4 | Observation / model / official kept apart | table stakes | ✔ (family ticks, words) | the forecast frames are set apart in the time bar |
| 5 | Impact-based legend | differentiator | ◐ (what it is *not*) | follow-up (needs agency-sourced effect bins) |
| 6 | Saved places with alerts (push or email) | retention | ◐ (home port, "since last check") | **follow-up**: needs a backend and a sender |
| 7 | Offline trip briefing | table stakes (fishing) | ✔ (field pack) | unchanged |
| 8 | Time slider with **playback**, outlook marked as such | table stakes | ◐ (steps, no play) | **play / pause, Space, forecast break** |
| 9 | Cloud and gap handling made visible | differentiator | ✔ (hatch) | the hatch is calmer at state scale, full strength near shore |
| 10 | Anomaly against climatology | differentiator | ✗ | follow-up (needs a climatology product) |
| 11 | In-situ evidence beside satellite | differentiator | ✔ (pier samples, Research) | unchanged |
| 12 | Structured citizen or fleet reports | differentiator | ✗ | follow-up (needs moderation) |
| — | **Tide curve with a now marker** (every tide app's first picture) | table stakes (fishing) | ✗ (a text line in a fold) | **tide strip in the briefing** |
| — | **Places with live numbers on the first screen** (Surfline, Apple Weather, Windy favourites) | table stakes | ✗ (a bare question) | **front door with measured readings** |
| — | A/B compare (swipe) | differentiator | ✗ | follow-up |

## 2. What makes the best of them beautiful

Patterns this pass judged to matter most for a map-first ocean app, with what Neritic does about each.

1. **The basemap recedes.** Felt: the basemap recedes, data in the middle, annotations forward; a few colours, each with one meaning. v7's land was a bright blue-grey picture with seams where the relief image ended. v8 makes the land graphite, matched to the relief, and quiets lakes, borders and labels.
2. **Perceptually uniform, variable-specific ramps** (cmocean `algae` for chlorophyll, on a log scale). Neritic already used a viridis-like log ramp for chlorophyll and a sequential one for probability. Unchanged; they are defined in the pipeline palette and covered by parity tests.
3. **One honest time model** (RainViewer 8.0): one readout, ticks that keep gaps visible, a clear break between observed and forecast frames, playback behind one button, Space to play. v8 adds play/pause, Space and the break.
4. **Smooth fields at overview, true cells up close.** Windy and Ventusky draw continuous fields. v7 nearest-sampled everywhere, which drew stair-steps at state scale. v8 smooths below zoom 8; from harbour scale every pixel is a source cell again.
5. **Restraint around the data.** RainViewer simplified before adding glass. Apple raised the opacity of its iOS 26 bars after legibility complaints. Decks over the map stay solid in v8 (no blur over WebGL), with a bezel highlight and layered shadow.
6. **Tabular, quiet numerals; data values never animate** (Flighty uses monospaced digits). Already the project rule.
7. **One authored motion moment** (impeccable, Anthropic's frontend-design). For Neritic that is the opening. The owner asked for it to play in full, so v8 extends it to about 5 s. It stays skippable and is shown once per device.
8. **Anti-patterns avoided:** rainbow ramps; linear chlorophyll; several clocks in the player; glass over data; silent interpolation across clouds; an outlook shown as live; autoplay without reduced motion; proportional numerals.

## 3. Tools evaluated

| Tool | What it is (verified 2026-10-22) | Used? | Why |
|---|---|---|---|
| **impeccable** (pbakaus/impeccable 4.7) | One skill with 24 commands plus 61 deterministic detector rules, as a CLI and hooks | **Yes: the detector**, run on `src/` and on the live pages (`npx impeccable detect <url>`). Its craft floor and audit checklists also guided the pass. | Findings and outcomes are in the commit "Craft floor". The skill's hooks were **not** installed: they would run in every session sharing this checkout. |
| **Vercel Web Interface Guidelines** (vercel-labs) | A MUST/SHOULD checklist; an agent skill that fetches it | **Yes: checklist pass.** Added touch-action, selection and caret, and the instrument scrollbar; confirmed color-scheme, theme-color, safe areas, no `transition: all`, reduced motion. | — |
| **Taste Skill** (Leonxlnx/taste-skill 2.1.4) | Classifies page vs app screen, plus an inspector and lint | **Rules read, not installed.** Its app-screen rules informed the cards: one border, pale tints, tabular figures, "a forecast is not a fact" (dashed after today, as the tide strip's extrapolated stretches are). | It bans violet, which is Neritic's *model* colour, and hairlines, which are the house style. Adopting it would need `--keep palette`. |
| **21st.dev MCP** (now "21st MCP", HTTP) | Component search and AI generation | **Not used.** | It needs an API key (all old Magic keys were reset; a free key comes from 21st.dev/mcp) and an AI subscription for generation. Owner action if wanted. |
| **shadcn/ui** (CLI v4.22, Base UI default) | Copy-in components on Tailwind v4 | **Not adopted.** | Neritic is already Tailwind v4 with its own token system and primitives. `shadcn init` would write `components.json`, `cn` and CSS variables that collide with `--nx-*`. Its pieces worth having (Base UI drawer, cmdk, sonner) can be taken one at a time if needed. |
| **Magic UI** | 80 animated components (shadcn registry) | **Not used.** | Most (beams, shimmer, gradient text, glows) are what impeccable and Taste flag as tells. `number-ticker` conflicts with "data values never animate". |
| **Spline** (`@splinetool/react-spline` 4.1) | 3D scenes in the browser | **Not used.** | The runtime is ~0.9 MB gzipped plus wasm, and a second WebGL context beside MapLibre, which is a battery and memory cost on phones at sea. The opening uses CSS and SVG from the mark's own isobaths instead. |
| **Motion** (v12 in repo, v14 current) | Animation | Already used | Upgrade later, separately. |
| **Others read:** Anthropic frontend-design, Emil Kowalski's skills, Rauno Freiberg's interaction notes, Base UI, React Aria, cmdk, sonner, vaul (unmaintained), cobe | | Guidance only | Durations and easing (fast feedback, ease-out, exit faster than entrance) match `theme.css`. |
