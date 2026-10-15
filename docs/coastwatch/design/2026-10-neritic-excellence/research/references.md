# Neritic design reference research (2026-10-15)

Screenshots are in this folder. They were captured with headless Chromium and each one was viewed. On GFW, the welcome modal was closed with its own close button.

## Screenshots

### Linear
- **linear-desktop.png:** Linear home. The real product UI is the hero; secondary text is mid-grey.
- **linear-phone.png:** the same page on phone, with a low-contrast grey subtitle.
- **linear-method.png:** the Method page, with a serif display headline.

### Global Fishing Watch
- **gfw-desktop.png:** the welcome modal on first load.
- **gfw-desktop-map.png:** each layer row holds its own legend and unit line ("hours / 32,000 km²"). Grid cells are visible, and the timebar is a stacked histogram of the data. There is also a promo card and "?" hotspots.
- **gfw-phone.png:** the welcome modal on phone.
- **gfw-phone-map.png:** the phone layer list, with a "LAYER LIST | MAP" switch.

### Copernicus Marine
- **copernicus-viewer-desktop.png:** each layer row shows date, depth and cadence. There is an inline ramp in °C, a vertical depth slider and a multi-year timebar.
- **copernicus-viewer-phone.png:** the layer panel takes about 45% of the screen. The date shown is tomorrow, with no forecast label.

### earth.nullschool.net
- **nullschool-desktop.png, nullschool-phone.png:** the only chrome is an "earth ≡" button. There is no legend, date or source.
- **nullschool-sst-desktop.png:** particles drawn over the sea-surface temperature layer.
- **nullschool-point-readout.png:** "35.29° N, 127.62° W / Ocean Currents | 200° @ 0.12 m/s / SST | 18.2 °C".
- **nullschool-menu-open.png:** a key/value table of Date (Local ⇄ UTC), Data, Source and Scale. The scale has no tick values.

## 1. Emil Kowalski (emil-design-eng SKILL.md, review-animations STANDARDS.md, "You don't need animations", Vaul)
- **Frequency decides motion.**
  - Used 100+ times a day: no animation.
  - Tens of times a day: reduce it.
  - Occasional (drawers, toasts): standard animation.
  - Rare: delight is allowed.
  - Never animate keyboard actions.
- **Curves.** Never use ease-in.
  - Enter or exit: (0.23,1,0.32,1).
  - Move on screen: (0.77,0,0.175,1).
  - Drawers: (0.32,0.72,0,1).
  - UI motion stays under 300 ms.
- **Origin.** A popover grows from its trigger. Never start from scale(0); use 0.95 with opacity 0.
- **Interruptibility.** Use transitions, not keyframes, for anything that toggles. Use springs for gestures.
- **Vaul drawer.**
  - 500 ms on the drawer curve.
  - A flick above 0.11 px/ms dismisses.
  - Damping past the edge.
  - Drag only once the content is scrolled to the top.
- **Tooltips.** The first is delayed; later ones open instantly. Reduced motion means fewer and gentler animations, not none.

## 2. Impeccable
- **Operate mode.** Scanability, consistency and the real usage scene outrank expression. Choose light or dark from the use scene, not the product category.
- **Borrowing from the subject's world.** It may lend type, palette, density and one signature move. It never supplies layout, navigation or controls, and no "costume" such as a chart-plotter bezel.
- **Type rules:**
  - body 16 px or larger;
  - line length 45–75 characters;
  - light-on-dark text gets more leading and tracking;
  - tabular figures for data.
- **Colour rules:**
  - build roles in OKLCH;
  - body text 4.5:1, controls and focus 3:1;
  - colour is never the only code;
  - design dark mode on its own rather than inverting light.
- **Refused patterns:** eyebrows, gradient text, decorative glass, a coloured left border thicker than 1 px, nested cards, sparklines standing in for content, emoji icons.
- **Motion:** feedback 100–150 ms, state changes 150–300 ms, overlays 300–500 ms. Exits are faster than entrances.

## 3. Taste Skill
The skill says itself that it is not for dashboards or app screens.
- **Tells registry:** violet gradients, glass, a shadow under every card, three identical icon cards, a neon glow, a pulsing "live" dot, cream with terracotta.
- **Proportion:** background 70–90% of the screen, accent under 5%. Surfaces separate by a tone step. No nested panels.
- **Its required motion budget is wrong for a tool.**

## 4. Linear
- **Themes:** 98 variables per theme reduced to 3 LCH inputs (base, accent, contrast). A contrast setting generates high-contrast themes.
- **Hero:** the real product as the hero, not an illustration.
- **Method:** "Aim for clarity: avoid inventing terms."
- **Weakness:** mid-grey on near-black.

## 5. Global Fishing Watch
- **Layer rows:** each row carries its colour toggle, source, unit line and a stepped legend with ticks.
- **Timebar:** it doubles as a histogram of data presence.
- **Cells:** honest native-resolution cells.
- **Phone:** the layer list and the map are separate views.
- **Weakness:** the first load stacks a modal, a promo card and hotspots. The rail has unlabelled icons.

## 6. Copernicus MyOcean
- **Metadata line per layer:** date, depth and cadence, with a ramp that has ticks.
- **One axis per dimension:** time and depth each get their own control.
- **Weaknesses:**
  - a forecast date is shown with no forecast label;
  - the panel takes about 45% of the phone screen;
  - rainbow ramps.

## 7. earth.nullschool
- **Chrome:** minimal.
- **Tap readout:** value and unit for every layer, and the units cycle when tapped.
- **Times:** local time with a UTC toggle.
- **Weaknesses:**
  - closed, it shows no date, source or legend;
  - the scale has no ticks;
  - particles run right up to the coast.

## Adopt for Neritic
1. A provenance line on every layer and evidence item.
2. Legends inside the layer row, with units and ticks.
3. A time control that shows data presence and gaps, with a "today" mark.
4. The forecast part of the time control labelled and hatched, never marked by colour alone.
5. C-HARM drawn as native 3 km cells that stop before the shore, labelled "No model data at the shoreline".
6. A tap readout giving value, unit and valid time, or "No data here".
7. Units that cycle on tap and are remembered.
8. Local time first, with UTC on request.
9. On phone, map and list as separate views, with a sheet that has snap points and scrolls before it drags.
10. Real data as the hero.
11. A palette generated from a few OKLCH inputs plus a contrast setting, giving a "Sun" theme.
12. A light, high-contrast default for the phone briefing.
13. Status stated in words and shape.
14. Plain names and no eyebrows.
15. No stacked onboarding.
16. Don't move learned controls.

## Reject or adapt
1. Hidden chrome. Valid time and source stay visible.
2. Particles over gaps or over C-HARM.
3. Grey on black.
4. GFW's density and hotspots.
5. Taste Skill's motion budget.
6. Rainbow ramps and a panel that takes half the phone screen.
7. Monospace for data. Use tabular figures in the text face instead.

## Motion rules
- **Curves:** out (0.23,1,0.32,1); in-out (0.77,0,0.175,1); sheet (0.32,0.72,0,1).
- **Durations:**
  - press 120–160 ms;
  - tooltip 125 ms, then 0 ms;
  - popover 150–200 ms;
  - state change 200 ms;
  - sheet 400–500 ms;
  - fly-to 800 ms or less.
- **Never animate:** keyboard actions, time scrubbing, numbers, notices, legends, data fade-ins that hide staleness.
- **Reduced motion:** keep opacity, drop translate and scale, use static arrows, and jump instead of fly.

## Skills
Install none. If one, Emil Kowalski's review-animations (Markdown only).
- **Impeccable:** its launcher downloads a binary and adds hooks.
- **Taste Skill:** not for app screens.
