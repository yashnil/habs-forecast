# Neritic v11: Open Water

A design reset for the whole product (Today, Explore, My Coast, Research and the shell), meant for `main`, not for one demo. It answers the owner's review of the pilot build: "real substance, but crowded, generic, AI-generated; a dashboard, not a product."

Evidence behind it: [research.md](research.md) (reference products) and [ia-audit.md](ia-audit.md) (where the crowding comes from, with counts). Before screenshots: [before/](before/).

## 1. What was wrong, in one paragraph

Every surface used the same texture: a caps micro-label, a hairline, 13–15 px grey text, a small chart, a grey caveat. Nothing was larger or quieter than anything else, so nothing led. Explanations were always open (about 330 words of them on Today). The same fact was said up to six times on a page. Explore wrapped its map in ten chrome regions. The front door (Today) had no map at all, and the "contours" behind its title were invented curves. Motion had been cut back to almost nothing. Cyan on near-black with mono numerals and outlined chips is also the default look of a generated dashboard.

## 2. The idea

**The ocean is the stage; everything else is staged on it.** One living map of real water, with real observed currents moving on it, sits behind or beside every primary page. Words come first and are few. Numbers are large where they matter and absent where they do not. Detail is one tap away, never deleted. The interface itself has no colour: hue belongs to data and to official status.

Five rules, each from the research:

1. **Sentence first, then one graphic.** (Apple Weather.) Every page and every section opens with one plain sentence. A section's glance layer is that sentence plus one graphic. Everything else is its detail layer.
2. **Order by what matters now.** (Apple Weather.) A closure or delay leads. A quiet fishery collapses to a line. Stale things say so in place and step back.
3. **Chrome has at most three fixed zones.** (MyOcean Light, Global Fishing Watch, savvy navvy.) What, on one edge. When, along the bottom. Tools, as bare icons. The legend is the title of the map and its readout.
4. **Colour belongs to the data.** (Linear, Ventusky.) Chrome is neutral ink on deep water. The four data families and the three status colours are the only hues on screen.
5. **Motion explains or arrives; it never decorates.** Particles move only where water moves (measured HF-radar currents). High-frequency actions cut or cross-fade. One thing moves at a time.

## 3. Honesty rules (unchanged, and tightened)

These are acceptance criteria for every change in this pass.

- Order of authority: official status, then measurements, then forecasts, then advice. Never a green "safe". The calmest state is "none on record".
- "Not verified" stays in words wherever an unreviewed official record is quoted: on the lede, on the status group header (computed: "7 of 7 not verified"), in each record's sheet, and in the shell's trust pill on every page. Mixed groups mark the exceptions per row.
- "None on record" never appears without "not a promise it is open" on the same line or directly under it.
- Observed, model, outlook and analysis stay distinguishable by word, by family colour and by position on a time axis (past left of now, future right).
- A stale value carries "(old)" or its age on the value itself. A model run whose days are all past is a **past run**, never an outlook: it is labelled as such and steps back.
- Chlorophyll is never toxin. The words "not toxin" stay on the chlorophyll legend and outlook at every width.
- Every value's source and time stay reachable within one tap, and a compact freshness cue stays visible.
- Real geometry or nothing: any line that looks like a contour or a coast is drawn from real data (`MARK_FIELD`, `isobaths.geojson`, the map). The five invented curves in the Today hero are removed.
- No count-ups, no interpolation between observed days, no tweened data values. A chart may draw itself in once; its numbers do not animate.
- Fix the contradictions the audit found (owners in §9): the wind join, the advisory end, the stale C-HARM run under "outlook", commercial status differing between Today and My Coast, the shellfish wording, two caveats hidden on phones.

## 4. Visual system

Token names keep the `--nx-*` prefix so existing components keep working. Values and a few global classes change in `src/v4/theme.css`.

### 4.1 Colour

| Role | Night (default) | Notes |
|---|---|---|
| Ground `--nx-bg` | `#04090e`, with a fixed page gradient from `#071722` (top) to `#03070b` | Deep water, slightly blue. Not flat black. |
| Surfaces 1–3 | `#0a141c`, `#101d27`, `#182a36` | Used for sheets, decks, controls. No borders by default. |
| Ink 1–3 | `#eef3f5`, `#b3c2cb`, `#8598a4` | Ink 3 only for tertiary text ≥ 13 px. |
| Lines | `rgba(200,220,232,0.10)` and `0.22` | Hairlines only between list rows and on axes. |
| **Accent `--nx-accent`** | **`#eef3f5` (ink)** | Selection, focus and active states are ink, not cyan. `--nx-on-accent: #04121c`. `--nx-accent-bg: rgba(238,243,245,0.10)`. |
| Link `--nx-link` (new) | ink, underlined | Links are underlined ink. No blue links. |
| Official / model / measured / conditions | unchanged | Amber, violet, teal, sky. Data only. |
| Status | `--nx-down` red (closed, delayed), `--nx-stale` amber (restricted, watch, old), `--nx-offline` blue-grey (none on record) | Shown as a filled tint tag plus a shape, never an outlined capital chip. |

Daylight mode (`html[data-daylight]`) keeps its job and its 7:1 values; its accent becomes its ink too. Both themes must pass axe AA.

### 4.2 Type

Families stay (Newsreader, IBM Plex Sans, IBM Plex Mono). IBM Plex Sans Condensed is retired. What changes is scale and role.

| Role | Class | Spec |
|---|---|---|
| Place and page title | `.nx-display` | Newsreader, opsz, 500. Hero: `clamp(44px, 7vw, 96px)`, line-height 0.95, tracking −0.015em. |
| Section title | `.nx-title` | Newsreader 500, 26/30 px desktop, 22/26 px phone, sentence case. Replaces caps labels above sections. |
| Lede | `.nx-lede` (new) | Newsreader 400, 22–28 px, line-height 1.25, max 34ch. |
| Hero number | `.nx-figure` (new) | Plex Sans 300, tabular, 40–64 px, tracking −0.02em. Unit beside it at 0.4em, ink-2. |
| Value | `.nx-value` | Plex Sans 400 tabular (no longer mono), 20–26 px. |
| Label | `.nx-caps` (restyled) | **Sentence case**, Plex Sans 500, 12.5 px, ink-2, no tracking, no uppercase. The class name stays; its look changes everywhere at once. |
| Body | — | Plex Sans 400, 15.5/24 px. Secondary 13.5/20 px ink-2. |
| Note | `.nx-margin` | Newsreader italic 14.5 px, ink-2. One short line beside what it qualifies. |
| Mono | `.nx-data` | Plex Mono only for coordinates, station ids and axis ticks. |

Rule of three: any one screen uses at most three text sizes above the fold besides the title.

### 4.3 Shape, surface, depth

- Radii: 8 (tags, small controls), 12 (buttons, inputs), 16 (decks, cards), 24 (sheets, top corners only on phones).
- **Canvas**: the map or an image. Full-bleed. Nothing boxes it.
- **Deck** (`.nx-deck`): a control cluster floating on the canvas. `rgba(7,14,20,0.9)`, radius 16, no border, shadow `0 1px 0 rgba(255,255,255,0.04) inset, 0 10px 30px rgba(0,0,0,0.35)`. No backdrop blur over WebGL.
- **Sheet** (`.nx-sheet`, new): a reading surface that rises over the canvas. Surface 1, radius 24 on its leading edge, generous padding (24–40 px).
- **Plate** (`.nx-plate`, restyled): surface 1, radius 16, no border. Never nested.
- Text sits directly on the ground wherever it can. A box needs a reason.
- **Tag** (`.nx-tag`, new): status word with its shape, filled tint, sentence case, radius 8, 13 px 500. `data-level="closed|delayed|restricted|watch|none"`.
- **Not-verified marker** (`.nx-unverified`, new): a small shield-slash icon and the words "Not verified", ink-2 with a dotted underline; opens the explanation. Never colour only.
- Buttons: primary is ink-filled (dark text). Secondary is surface 2 filled, no outline. Tertiary is underlined text. Minimum 44 px, 56 px for primary actions on phones.
- Whitespace: sections are separated by 72–112 px on desktop and 48–64 px on phones, not by rules.

### 4.4 Ambient graphics

- The real Monterey Canyon isobath field (`MARK_FIELD`) is the brand's ground: at 5–8% ink on quiet page heads (My Coast landing, Research head, legal pages, the offline poster). It is the brand mark's own geometry, not a locator map, and is never labelled as the reader's waters.
- No gradients for decoration, no glow, no noise textures, no glass.

## 5. Motion system ("Swell")

Existing tokens that tests pin keep their values (`--nx-t-press` 90 ms, `--nx-t-release` 220 ms, `--nx-t-layer` 220 ms, `--nx-t-page` 200 ms; raster fade 220 ms; panel travel 12 px). New tokens:

| Token | Value | Use |
|---|---|---|
| `--nx-t-reveal` | 480 ms | A section entering view (opacity, 12 px rise). |
| `--nx-t-draw` | 700 ms | A chart drawing itself in, once. |
| `--nx-t-stage` | 1400 ms | The stage arriving (camera settle). |
| `--nx-t-route` | 320 ms | Route cross-fade (View Transitions). |
| `--nx-ease-emph` | `cubic-bezier(0.05, 0.7, 0.1, 1)` | Entrances that travel. |
| `--nx-ease-exit` | `cubic-bezier(0.3, 0, 0.8, 0.15)` | Exits; a third shorter than the entrance. |

Primitives (script in `src/v4/motion/`, CSS in `src/v4/motion/motion.css`):

| Primitive | What it does | Rule |
|---|---|---|
| `Reveal` | Fades and rises a block the first time it enters view. | Chapter-level blocks below the fold only. Never above the fold, never per row. Stagger only inside one group, 50 ms, at most 5 items. |
| `DrawOn` / `useDrawOnce` | Lines draw left to right, bars and areas grow from the baseline, marks fade in after. | Once per chart, only when first seen. Values and labels do not tween. |
| `Stage arrival` | The map eases in from 0.5 zoom out while the title, then the lede, then the glance row rise in three beats. | Once per session per port. Later visits open settled. |
| `Scroll-linked hero` | As the briefing sheet rises, the stage dims to 55% and scales 1.04, and the place name condenses into the top bar. | CSS `animation-timeline: scroll()` behind `@supports`; static otherwise. |
| `Sheet` | Side sheet on desktop (420 px, slides 24 px), bottom sheet on phones (drag to dismiss, the one spring). | Focus is trapped and returns. Escape closes. |
| `Fold` | Expand in place (grid-rows), 240 ms. | The words stay in the DOM. |
| `Route transition` | `document.startViewTransition` around app-router navigation: content cross-fades 320 ms; the top bar holds still (`view-transition-name: nx-topbar`); the stage morphs between pages (`nx-stage`), so Today's hero map grows into Explore. | Feature-detected. Reduced motion: instant. |
| `Glide` | The nav pill and any segmented control's thumb slide to the new item, 260 ms. | Existing `useGlidePill`, generalised. |
| `Hover` | Rows tint 6%; cards lift 1 px; the linked thing on the map lights (a fishery row lights its notice line; a test row lights its site). | Pointer devices only (`@media (hover: hover)`). |
| `Press` | `scale(0.97)`, 90 ms. | Every pressable. |
| `Layer switch` | Raster cross-fades 220 ms; the legend swaps with a 120 ms fade; a day step cuts. | Pinned by tests. |
| `Playback` | The time rail can play the frames it has (C-HARM days, satellite days) at one frame per 900 ms with a pause control; each frame is labelled with its own date and kind. | Stops on any input. Never interpolates between frames. |
| `Currents` | Measured HF-radar surface currents drawn as moving streaks (`FlowLayer`). | Only where there are observed cells. One still frame under reduced motion. Pauses off-screen and when the tab is hidden. |
| `Loading` | Nothing for 300 ms, then a flat skeleton that mirrors the final layout, or the real-isobath `ContourLoader`. | No shimmer. Minimum visible time 300 ms once shown. |

Reduced motion (`prefers-reduced-motion`): fades stay, nothing travels, nothing loops, the camera jumps, currents are a still frame, route changes are instant.

What is deliberately not animated: numbers, status changes, layer and day switching beyond the cross-fade, anything triggered from the keyboard at high frequency.

## 6. The stage (`OceanStage`)

One component, `src/v4/stage/OceanStage.tsx`, wraps `MapCanvas` for use as a scene outside Explore.

- Loads lazily after first paint (`next/dynamic`, `ssr: false`) when it is near the viewport. Before that, and whenever the map cannot run (offline pack, save-data, WebGL failure), it shows the **poster**: the real isobath field on the deep ground, inline SVG, zero requests.
- The text over it is server HTML and never waits for the map.
- Props: `center`, `zoom`, `pitch` (0–50), `bearing`, `field` (`"chlorophyll" | "charm" | "outlook" | "sst" | "none"`), `currents` (boolean), `marks` (ports, test sites, buoys), `lines` (notice and RAMP geography), `focus` (which mark or line is lit), `interactive` (false by default: the page scrolls through it), `caption` (the legend-as-title: what, kind, date, "not toxin" where it applies).
- A scrim (ground colour, 0 → 85% at the bottom) keeps overlaid text at AA contrast over any pixel.
- Only one `MapCanvas` per page (`window.__cwMap`).
- A pitched camera is allowed here (not in Explore, which stays north-up and flat for reading). If `FlowLayer` or the rasters misdraw under pitch, the stage falls back to pitch 0.

## 7. Pages

### 7.1 Shell

- One frame for every page: `TopBar` (brand, four workspaces with the gliding pill, one **trust pill**, bell, more) and the phone dock. `PilotShell` pages and v4 pages share it.
- The top bar is transparent over a stage and takes a solid ground once the page scrolls.
- **Trust pill**: "Research preview · 3 notices here · not verified", with a clock when any source is old. It replaces the separate disclosure line and the notices cluster. It opens the notices and data status. It is on every page, Today included. Test ids `registry-disclosure` and `notices-button` live on it.
- The daylight switch lives in one place (the more menu, and a sun button on phones). The hero copy is removed.

### 7.2 Today (the front door)

Above the fold, in order:

1. **The stage**: the reader's waters, 68svh on desktop and 52svh on phones, field = chlorophyll (observed, dated, "not toxin") with measured currents moving on it; the port ringed; notice geography drawn.
2. Over its lower left: date line; **place name** (also the port switcher); **the lede**, one sentence from the top official fact, with its own "CDFW · checked Nov 8 · Not verified"; one sea line ("Tonight N 5–10 kt, seas 6–9 ft · small craft advisory").
3. Along the bottom of the stage: the **glance row**, four unboxed items, each a link to its chapter and each lighting its subject on the map on hover or focus:
   - Fisheries: the seven-mark ribbon and "4 flagged";
   - Sea: now and the next change;
   - Tests: "5.5 of 30 ppm · Sep 28 (old)";
   - Bloom: "about usual · next 8 days".
4. A "since your last visit" line only when something changed.

Then the briefing rises over the stage as a sheet with chapters, single column, generous spacing, each a sentence plus one graphic with detail behind a fold or a sheet:

| # | Chapter | Glance | Detail |
|---|---|---|---|
| 1 | Fisheries (official status, with whale rules) | Group header with the computed not-verified marker. Flagged fisheries as large one-line rows with tag and cause. One collapsed line: "None on record for rock crab, groundfish, market squid. Not a promise they are open." | Row opens the sheet: cause, area, next review, source, checked date, review state, RAMP zone ladder. |
| 2 | Sea | Advisory band, three figures (wind, waves, water) and one 36-hour strip with the tide under it on the same axis and a now line. | Fold: all NWS periods, buoy provenance, tide table. Link: "5 days against your limits" → My Coast. |
| 3 | Toxin tests | One sentence and one shared 0–limit dot strip per species (dots against the action level). | Fold: per-site sets, key, CDPH link. |
| 4 | Bloom | Two labelled steps on one time axis: NOAA C-HARM (next 3 days, or "past run" when stale) then the Neritic 8-day chlorophyll outlook against usual. | Fold: what each is, skill, caveats. Link: "See it on the map" (morphs into Explore). |
| 5 | Keep, alert, ask | Three actions. | Readiness text inside each flow. |

Removed: the "At a glance" aside, the hero sun toggle, the first-visit paragraph, four of five feedback prompts (one remains at the end), the whale section as a separate chapter.

No port chosen: the stage shows the whole coast with ports marked; "Where do you fish?" with search and the pilot ports. Picking a port flies the camera there (the descent) and the briefing rises.

### 7.3 Explore

Three zones around a full-bleed, north-up map:

1. **Top**: the shared bar with search.
2. **Field dock**, bottom left: one deck. Its head is the legend-as-title ("Chlorophyll · observed · Nov 7 · not toxin", the ramp, and a readout marker that moves along the ramp under the cursor or the selected point). The layer list opens from it, grouped by kind: Observed, Model, Outlook, Reference. Beside it, the **time rail**: one track, past left of now, future right, data ends where it ends, with play.
3. **Tools**, right edge: zoom, locate, and one "more" (share, field pack, select area).

A place opens a **place sheet** (right on desktop, bottom on phones) led by its name, three statements in authority order, and "Open today's briefing"; evidence detail is behind folds. Onboarding hints become a one-time coach mark. Credits sit in a 20 px margin line. Target: about 84% of a 1440×900 viewport is unobstructed map at rest (from 67%).

### 7.4 My Coast (one trip)

- Landing: the question, the search, and the coast itself as the selector. The six-card readings grid becomes a short list of names (home port first); readings show on hover or focus of a point.
- Briefing: Today owns the port now; My Coast owns one trip. Above the fold: title; the **trip line** ("Tue Nov 10 · Dungeness, commercial · Change"); the status line for that species from the same model as Today, with its not-verified marker and "Nothing here says a trip is allowed" at every width; three figures (wind, seas, tide); the **60-hour lanes** against the reader's limits. The stage sits beside it on desktop and above it on phones.
- **Trip ledger** (economics): a designed module, not a form. Inputs on the left in plain rows, a live waterfall on the right (gross, then each cost, then crew share, then net) that redraws on every keystroke without animating numbers; scenarios as tabs; "hypothetical" always shown.
- Checklist opens as a sheet. The source strip and the duplicate notices list are deleted (their content is in the lanes and the claims).

### 7.5 Research

- The record chart is the page. The always-open 17-station ladder becomes a station picker opened from the title or the mini map.
- Header readouts fold into the crosshair's default reading. Notes become two info popovers.
- Chart style: no chart borders, axis ticks in mono ink-3, gridlines only at decades, panel titles in sentence case carrying "measured at the pier" against "model probability", direct labels instead of a legend where possible, draw-in once.

## 8. Performance and degradation

- Server HTML carries every word of the answer. The stage is progressive.
- No new runtime dependency.
- The Today offline pack must still open: offline it shows the poster, not a broken map. Any new static asset the offline page needs is inlined or added to the pack.
- `FlowLayer` keeps its own frame budget and pauses itself.
- No horizontal scroll at 320 px. Hit targets ≥ 44 px.

## 9. Build plan and ownership

| Track | Branch | Owns |
|---|---|---|
| F1 foundation: system and shell | `v11/system` | `src/v4/theme.css`, `src/app/globals.css`, `src/app/layout.tsx`, `src/v4/shell/**`, `src/pilot/shell/**`, `src/v4/ui/**`, shell tests |
| F2 foundation: motion and stage | `v11/motion` | `src/v4/motion/**` (new; `src/v4/motion.ts` re-exports), `src/v4/stage/**` (new), `src/v4/charts/kit*` (new), route transitions, minimal additive props on `MapCanvas` |
| T Today | `v11/today` | `src/pilot/today/**`, `src/app/today/**`, `src/pilot/facts.ts` wording fixes, Today tests |
| E Explore | `v11/explore` | `src/v4/explore/**`, `src/components/map/**`, Explore tests |
| C My Coast | `v11/coast` | `src/v4/coast/**`, `src/app/coast/**`, My Coast tests |
| R Research | `v11/research` | `src/v4/research/**`, `src/v4/charts/**` (except the kit), Research tests |
| Integration | `feat/neritic-v11` | merges, cross-page consistency, screenshots, the full suite, this document |

Honesty fixes by owner: wind join, advisory end, stale C-HARM labelling, lede marker, none-on-record caveat (T); commercial status from the shared model, shellfish wording, the phone caveat (C); "not toxin" on the phone peek (E).

## 10. Deferred on purpose

See [README.md](README.md) once the pass lands.
