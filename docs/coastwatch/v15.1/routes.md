# v15.1 route audit: every reader-facing address and its interface

Audited 2026-11-13 on `feat/neritic-v15` (4b0ecd875) and moved in branch `v151/d`. "Before" is what the route
rendered at the start of v15.1; "after" is this branch. Shell = the v14 route group `src/app/(neritic)` (one bar,
menu, tab bar and one persistent MapLibre chart, `src/nx/shell`). Info layout = `src/nx/pages/InfoPage.tsx`.

| Route | Reached from | Before | After |
|---|---|---|---|
| `/` | brand mark, start | shell (home / opening / Today) | unchanged |
| `/today` | nav | shell, stage | unchanged |
| `/explore` | Today, menu | shell, stage | unchanged (agent A owns the entry points) |
| `/coast`, `/coast/trips` | nav | shell | unchanged |
| `/research` | nav | shell | unchanged (agent C) |
| `/alerts` | nav | shell | unchanged |
| `/notices` | Today, Alerts, registry line | shell | unchanged |
| `/ask` | menu | shell | unchanged |
| `/account` (Your data) | menu | shell, desk | unchanged |
| `/about` | menu, footers | **v4 dark editorial** (`AppShell`, v4 TopBar, trust pill) | shell, info layout, contents list |
| `/sources` | menu, footers, Research errors | **v4 dark** (`AppShell`, `SourceTable`, `OfficialStatusCard`) | shell, info layout; every source row kept (`SourceRows`) |
| `/privacy` | menu, footers, Your data | **v4 pilot shell** (`LegalPage`/`PilotShell`) | shell, info layout (legal), contents list |
| `/terms` | footers, Privacy | **v4 pilot shell** | shell, info layout (legal), contents list |
| `/sms-terms` | footers, Terms | **v4 pilot shell** | shell, info layout (legal), contents list |
| `/fisheries` | Research (“Fisheries history”), next steps, About | **v4 dark** (`AppShell` + M7 components) | shell, info layout (wide); M7 components drawn through the token bridge (`src/nx/pages/bridge.css`) |
| `/coast/kept` | My Coast “Keep a dated copy”, offline page | **v4 dark** (`KeptView`, own frame, no nav) | shell, info layout (`KeptBriefing`); print hides the shell |
| `/account/pilot` | Your data (only where accounts are connected) | v4 pilot shell; 404 without Supabase | shell, info layout (wide) + token bridge; still 404 without Supabase |
| `/bloom` | old links | 308 → `/research` | unchanged (redirect, no interface) |
| unknown address | — | **Next.js default 404** (unstyled, outside any shell) | `src/app/not-found.tsx`: the same bar, menu and tab bar (`StandaloneShell`, no chart), info layout. A catch-all inside the group would keep the chart mounted, but it makes `@next/next/no-html-link-for-pages` flag every plain `<a>` in the app, so it was not used. `notFound()` inside the group (`/account/pilot`) renders `(neritic)/not-found.tsx` with the chart |
| error in a page of the group | — | **Next.js default error** | `(neritic)/error.tsx`: shell stays, info layout, “Try again” |
| error outside the group | — | Next.js default | `src/app/error.tsx`: same shell without the chart (`StandaloneShell`) |
| root layout failure | — | Next.js default | `src/app/global-error.tsx`: own document in the ds tokens and Figtree |
| `/system` | not linked (review surface) | v12 design-system catalog in the v4 motion root | unchanged: the catalog is the design system's own reference page |
| first visit (`/` fresh) | — | shell; cinematic opening (skip, reduced motion, static tier) | unchanged concept; see below |

## States inside the moved pages

- **Data unavailable**: every info page says “Neritic’s data could not be loaded…” (words from the old AppShell) above
  the registry line when the manifest cannot be read; Sources says the status cannot be shown; Fisheries keeps its
  “unavailable” note with the official sources.
- **Registry line**: the v4 trust pill's disclosure (research preview, N official notices in CA, Not verified, “Test
  data” on fixture data) as one line that opens in place: the not-verified statement, CDFW and CDPH, each current
  notice by name, a link to every notice on `/notices`, the hotlines and “No notice listed here does not mean an area
  is open…”. Test ids kept: `notices-button`, `registry-disclosure`, `notices-panel`, `fixture-banner`.
- **Kept briefing, none kept**: “No briefing is kept on this device…”.

## First visit and onboarding

The opening (`src/nx/opening/Opening.tsx`) and its pre-paint decision (`src/nx/shell/entry.ts`) were already on the
v14 system: dithered canyon still → isobaths ink → tints flood → neatline closes, ≤ 5 s, Skip and any input end it,
never under reduced motion, on a deep link or on the static tier. Not redesigned. The only onboarding control outside
the shell was About's “Set up for” (`src/components/welcome/PurposeSwitch.tsx`); it now draws in the ds tokens.

## Words moved verbatim (nothing dropped)

Footers (AppShell, PilotFooter), the data-unavailable sentence, the legal frame (draft banner, effective date,
CC BY attribution) and the kept briefing's words are in `src/content/pages.ts`; the three policies' bodies in
`src/content/legal/*.tsx`. About and Sources read `src/content/copy.ts` as before.
