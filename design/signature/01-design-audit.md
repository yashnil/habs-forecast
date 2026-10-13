# 01 — Design audit: CoastWatch's visual and interaction experience

**Date:** 2026-10-13. **Scope:** `main` (ce68a541) for history, and the in-progress redesign (`feat/coastal-redesign`, draft PR #41, 8dec135a) as the primary reference.
**Evidence:** the redesign's own screenshots (`docs/coastwatch/redesign/img/` on that branch: `before-*` is `main`, `after-*` is the redesign) and my reading of its source. I only read the other session's worktree; I changed nothing in it.

**Relationship to the redesign's product audit** ([`31-product-audit.md`](https://github.com/yashnil/habs-forecast/blob/feat/coastal-redesign/docs/coastwatch/31-product-audit.md)): that document already diagnoses the *product* problems and the redesign is fixing them: no synthesis, caveats repeated, a model-dominated map, research vocabulary, missing ocean conditions. I don't repeat those findings. This audit covers what is left once the information architecture is right: identity, visual hierarchy, motion, geography, loading and mobile ergonomics.

This is a design audit. No stakeholders were tested. "As a fisherman" below means a structured walkthrough from that point of view, not user research.

---

## What the redesign gets right — keep and build on

1. **The briefing order.** Place and freshness, then official status, next steps, conditions, and details. This is the right spine, and the prototypes keep it.
2. **Basemap first.** A depth-tinted seafloor when no data layer is on gives a calm, geographic first impression. Monterey Canyon is beautiful, and it is real.
3. **Official limits drawn as lines.** For example, "36°31.46′ N · south limit". This is the most useful graphic in the product, and it is the seed of the visual grammar proposed here.
4. **Honest gap language.** "No forecast for these waters", "No clear view this week (cloud, fog or no overpass)" and "nearest value 6.6 km NW (not a forecast here)".
5. **The settled semantic dots.** Official amber, conditions teal, measured green, model violet, gap grey. The prototypes reuse these names.
6. **Plain-language Bloom page.** "Toxin (domoic acid) in seawater 0.2 ng/mL · 12 days ago" is a model for every value in the product.

## Prioritised critique

Severity is the impact on comprehension and trust for a first-time stakeholder; it is not about polish.

### P1. Amber means everything, so it means nothing

**Seen in:** `after-coast-monterey-1440`, `after-bloom-1440`, `after-fisheries-390`.

Amber or cream is used for:
- the site-wide warning bar;
- the masthead notice pill;
- every official card's background and border;
- the "Not confirmed" freshness text;
- the Bloom page's notices banner;
- Fisheries' "None in CoastWatch's list" card.

On a typical first screen, 25–35 % of the paper is cream with amber rules. The colour that should mean *an agency said this* also means *warning*, *stale* and *nothing found*. Readers learn to skip it, which is the same failure the product audit found for repeated caveats, now carried by colour instead of words.

**Fix (prototyped):** reserve amber for official *content*: the rule line, the agency name and the notice title rule. The status line lives inside the navy masthead with a 3 px amber left rule; it is not a second cream bar. Freshness words use ink and grey, except "Not confirmed", which is a source-health fact. Cards lose their tinted backgrounds. See `#/system` and `#/explorer`.

### P2. On a phone, the answer starts below the fold

**Seen in:** `after-coast-monterey-390`, `after-coast-santa-cruz-320`.

At 390 × 844 the masthead (≈ 84 px) and the cream status bar (≈ 80 px) take 19 % of the screen. Then come the page title, the search field, "Make … my home port", four activity chips, the species select and four date chips. The first official notice starts at ≈ 830 px, so a fisherman sees nine controls before one fact.

At 320 px the "Not verified" pill wraps to two lines and "OFFICIAL NOTICES · THESE WATERS" wraps as tracked capitals.

**Fix (prototyped):**
- One masthead with the status line inside it, ≈ 100 px in total.
- A one-sentence answer (Level 1) directly under the place name.
- The plan controls (activity, species, day) as one row *after* the answer, because they refine the answer and are not a gate before it.
- On the map, the answer sits in the sheet's peek state.

### P3. Map controls in four corners, and three separate maps

**Seen in:** `after-map-coastal-1440`, `after-coast-*`, `after-bloom-*`.

The Ocean Map has:
- a left icon rail (Places, Forecast, Satellite, Currents, Tour);
- a region pill top-left;
- a layer card bottom-left;
- zoom and attribution controls bottom-right;
- the masthead mode switch top-right.

The eye travels to every corner. My Coast, Bloom and the Ocean Map each create their own map at a different zoom and style, so moving between them loses the reader's place: a geographic product with no geographic continuity.

**Fix (prototyped):**
- Two control zones: *where* (search, layers, centre) at the top and *when* (time) at the bottom. One panel on the left.
- Coastal journeys animate between places along the coast.
- **Depends on the redesign:** a single persistent map instance shared across routes, which keeps camera continuity between Map, My Coast and Blooms.

### P4. The forecast field still dominates when shown

**Seen in:** `after-map-forecast-1440`.

The basemap-first default fixes the first impression. Turning the forecast on still floods the bay with opaque pink, because on most days C-HARM is uniformly 60–90 %. The coastline, isobaths and labels sit under the raster, and the legend's caveat ("10-point colour steps for display, not risk levels") is a sentence the reader has to decode.

**Fix (prototyped):** three compositing modes, none of which changes a value or a cell edge.
- **Veil:** class colours pre-mixed toward the sea; isobaths, coastline and labels drawn above.
- **Lens:** full strength only in the cells around the chosen point.
- **Class lines:** probability classes drawn as "isobaths" along exact cell edges.

The swap between days dips through the basemap and never blends two days.

### P5. Uppercase, tracked eyebrows everywhere

**Seen in:** `MY COAST`, `OFFICIAL NOTICES · THESE WATERS`, `SHORE STATION · MONTEREY BAY · CALHABMAP`, `MEASURED AT THIS PIER · ABOUT WEEKLY`, `CONTEXT FOR PLANNING · NOT A TOXIN OR FISHING FORECAST`.

These are 11–12 px tracked capitals in a coloured ink. They are the most frequent typographic element on every page, they wrap badly on phones, and some carry critical meaning in the hardest-to-read style. "NOT A TOXIN OR FISHING FORECAST" is a rule the redesign must keep, and it is set in its weakest type.

**Fix (prototyped):** sentence-case kind labels led by the matching mark glyph, for example "▪ Modelled" or "— Official". Disclaimers that change meaning become normal-weight sentences in the body. The settled card pattern ("uppercase eyebrow + serif headline") is respected in spirit (a kind label plus an answer headline), with a proposal to drop the uppercase.

### P6. The serif headline sometimes states a non-answer

**Seen in:** Fisheries, "None in CoastWatch's list" set in 24 px Newsreader. My Coast at 320, "No notice in CoastWatch's list".

The settled pattern ("serif headline that states the answer") is strong. When the answer is an *absence*, though, the big serif reads as an all-clear, even with the caveat line under it.

**Fix (prototyped):**
- Absences are phrased as a statement about limits: "No notice's official limits include these waters", with "That is not the same as safe or open" on the next line in the same block.
- They carry the gap mark, not the official mark.
- They are never set larger than a positive finding would be.

### P7. Green "Current" next to a toxin value

**Seen in:** `after-bloom-1440`, "● Current 12 days ago" (green dot), directly above "0.2 ng/mL".

Freshness is shown with a green dot, and green is also the measured colour. Placed beside a toxin reading, green is the colour most likely to be read as "fine". The hard rule is no green all-clear states; this is not one by intent, but it is one by proximity.

**Fix:**
- Freshness words never take a colour of their own. Ink means current, grey means stale or historical, and the official ink means "Not confirmed" (a source-health fact).
- The measured mark is a *dot that carries its date*: filled when sampled in the last 14 days, a ring when older. Age is encoded in the mark's shape, not its colour.

### P8. Next steps are a wall

**Seen in:** `after-coast-research-1440` (right column).

Eight steps, each with a bold title, a two-line explanation, a source link and a coloured bullet, add up to about 300 words. Each step is right; together they are a document.

**Fix:** this is Briefing.tsx, which the primary session marked as in flux, so I did not redesign it. The ledger pattern below shows a possible shape: the step title as a row, the reason and source on expand, and the source link as a button.

### P9. No motion language, so state changes are invisible

**Seen across:** layer toggles, date chips, mode switch, place changes.

- Layers appear and disappear instantly.
- Date changes swap the raster with no cue that anything changed, so the reader can't tell whether the 60–70 % they see belongs to today or to the old day.
- Place changes jump-cut the camera.
- There is no visual link between the panel's statements and the map features they describe.

**Fix (prototyped):** a small, strict motion system.
- Dip-through-basemap data swaps (100 + 140 ms).
- Provenance threads that draw once from the point to each source.
- Coastal journeys you can interrupt.
- Sheet settling with a CSS `linear()` spring.
- All spatial motion collapses to 120 ms fades under reduced motion.

### P10. Mobile navigation has six tabs

**Seen in:** all `after-*-390` (Map, My Coast, Blooms, Fisheries, Notices, About).

About is not a destination people switch to while working, and six 65 px targets at 390 px leave no room for labels at 320 px.

**Fix (prototyped):** five tabs (Map, My Coast, Blooms, Fisheries, Notices). About moves to the masthead menu. Drawn icons replace the generic ones, and Notices uses the official rule glyph.

### P11. Loading and failure have no designed states

**Seen in:** source reading on `main`.

The code handles failures carefully (each part "fails on its own and says why"), but visually:
- loading is "Reading…" text or nothing;
- placeholders do not match final layout, so the page shifts;
- the map shows a no-data hatch under a raster that hasn't arrived, so for a moment the whole sea reads as "no data".

The last point was found and fixed in the prototype, where it happened too.

**Fix (prototyped):**
- The final layout is reserved from the first frame.
- Each ledger row has its own loading words after 400 ms.
- The basemap stays calm until a field exists.
- There are seven named states: loading, current, not confirmed, stale/historical, unavailable, no value, saved copy.
- A priority order guarantees the official links before any data. See `#/states`.

### P12. Charts are correct but inert

**Seen in:** the Bloom station chart on `main` and the redesign.

Switching quantity or window re-renders from scratch. Notices are not visible on the timeline, so the relation "the advisory came after this rise" requires two pages.

**Fix (prototyped, `#/station`):**
- Samples keyed by time move to their new positions.
- Notices are amber rules on the timeline.
- Gaps are hatched.
- The model sits on its own labelled axis.
- A five-step story is computed from the data.

---

## The persona walkthroughs (first 10 seconds, redesign `after-*`)

| Persona | What they need first | What they see first | Gap the design (not the IA) leaves |
|---|---|---|---|
| Commercial fisherman, Monterey, phone | Does anything restrict my catch today, and what's the sea doing? | Masthead, cream bar, title, nine controls | The answer is below the fold (P2) |
| Shellfish harvester, Humboldt | Is there an advisory for clams at my beach? | A good notice card, but in amber like everything else | Hierarchy: official vs warning vs stale (P1) |
| Researcher | The cell value, its date and its provenance | Research mode opens sections; the map cell box is present | No visual link from a value to its cell, run and station (P9) |
| Coastal operator | Conditions and any notice near the port | Conditions grid, deep in the page | Density (P8) |
| General visitor | What is this, and is the sea "bad"? | Calm seafloor map and a purpose card | Turning on the forecast floods the map (P4) |

## What to remove

- The cream site-wide bar as a separate band. Fold it into the masthead.
- Tinted card backgrounds for official content. Use a rule line and the agency name instead.
- The left icon rail on the map. Layers become one segmented control.
- About as a mobile tab.
- Tracked uppercase eyebrows.
- Map instances per page, eventually (depends on architecture).
