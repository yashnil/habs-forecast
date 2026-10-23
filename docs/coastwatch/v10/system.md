# Neritic v10: the Ocean Instrument system

Direction A of [the concepts](../design-research/concepts.md), chosen by the lead. It takes two things from the other directions, folded into this one system rather than shipped as separate themes:
- **From B (Maritime Atlas):** the editorial voice. A serif display for places and headlines, and caveats set as margin notes.
- **From C (Coastal Mission Control):** shared time lanes, used for the My Coast decision view.

Reference renders: `scratchpad/concepts/renders/a-*.png` (the A set) and `concepts/a-instrument/*.html`.

## Rules
1. **The ocean is the interface.** Maps run full-bleed wherever geography matters. Reading surfaces are plates over the same dark ground, never a second light theme. Daylight mode (`html[data-daylight]`) is the high-contrast choice for bright sun.
2. **One primary thing per screen.** Explore: the field. My Coast: the trip decision. Research: the chart. Everything else is one step away, through progressive disclosure that keeps its words in the page.
3. **The four data families keep their colours everywhere:**
   - official `--nx-official` (amber);
   - model `--nx-model` (violet);
   - measured `--nx-measured` (teal);
   - conditions `--nx-conditions` (sky).

   Families never blend, state is never shown by colour alone, and a gap is never drawn as a low value.
4. **Type roles:**
   - `.nx-display` (Newsreader) for places and headlines;
   - `.nx-caps` (Plex Sans Condensed small capitals) for labels;
   - `.nx-value` (Plex Mono light) for readings, with units small beside them;
   - Plex Sans for interface and reading text;
   - `.nx-margin` (Newsreader italic) for caveats and context, set beside the thing they qualify, not at the bottom of the page.
5. **Plates, not cards:**
   - `.nx-plate` gives one hairline and square shoulders (4 px). Nothing is nested inside cards.
   - Pills are for chips only, never for layout.
   - Rows are separated by hairline rules (`.nx-rule`).
6. **Every value carries its time and source** next to it: a caps label plus a one-line provenance. Stale and failed states appear in place.
7. **Motion explains state:**
   - a layer switch fades in 220 ms; a day step cuts;
   - selection draws the true footprint;
   - charts scrub with a crosshair;
   - economics updates on every keystroke without animating numbers;
   - panels move 12 px with ease-out; nothing overshoots;
   - reduced motion gets a still version.
8. **The guardrails** in [integrity-guardrails.md](../design-research/integrity-guardrails.md) are non-negotiable, and its checklist is the acceptance list.

## Ownership (v10 build)
- **Lead:** tokens (`theme.css` top and the v10 vocabulary block), fonts (`app/layout.tsx`), `TopBar`, `MotionRoot` and `Splash` (the opening), and integration.
- **Module owners:** each page's composition inside its own folder. They do not edit tokens, global styles, `TopBar` or other pages.
