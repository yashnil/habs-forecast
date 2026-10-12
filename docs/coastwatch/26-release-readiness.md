# 26 — Release readiness

**Status (2026-10-12):** M8 is merged (`main` 69d0d6e8). The full website exists only as a private Vercel preview behind a login. The public Research Preview (coastwatch-demo) is unchanged. **The full website is not ready for a public release** until every blocker in §1 is closed.

Working materials:
- [Official-notice review worksheet](release/notice-review-2026-10-12.md) (blocker B1)
- [Independent scientific review: request draft and reviewer checklist](release/scientific-review-request.md) (blocker B2)
- [Proposal: station observation CSV export](release/station-csv-export-proposal.md) (not a blocker)

## 1. Blockers (all must be closed before any public release)

| # | Blocker | Who | How it is closed |
|---|---|---|---|
| B1 | **Official notices have never been checked by a person.** The 10 records were transcribed by an AI assistant; issue #12 has stayed open since 2026-10-10. | Owner | Work through the [worksheet](release/notice-review-2026-10-12.md), edit the registry where needed, then run `cwp review-official --reviewer "<your name>" --confirm`. Close #12 only after the next watcher run matches. |
| B2 | **No independent HAB scientist has reviewed the application.** This covers how C-HARM is described, how station observations are read, the fisheries species tiers and the scientific wording. | External reviewer | Send the [request](release/scientific-review-request.md); record the findings and the changes made. The About page's "Review status" is updated only after the review has happened. |
| B3 | **No real screen-reader test.** axe and the accessibility tree are clean (0 violations in 16 states, doc 25 §5), but no person has used the site with VoiceOver. | Owner or tester | Do the [VoiceOver pass](#3-voiceover-test-script) on macOS Safari and iOS Safari, and fix anything that blocks a task. |
| B4 | **At 320 px wide the masthead overflows by 28 px.** The page scrolls sideways and the notices pill is clipped at the right edge ([screenshot](release/w320-map.png)). WCAG 1.4.10 (Reflow) requires 320 px. | Claude, small fix | Let the masthead pill shrink (count only) below 360 px; add 320 px to the phone tests. |

## 2. Should be done before a public release (not blocking)

| # | Item | Note |
|---|---|---|
| S1 | **The status line wraps at 375 px.** | Checked 2026-10-12, **acceptable**: it wraps to 2 lines (46 px instead of 35 px). "Notices not verified. Check CDFW and CDPH." and the More button stay fully visible. More opens the full reason and the link to the list, and nothing is clipped. No page scrolls sideways at 390, 375 or 360 px ([375 px](release/w375-map.png), [More open](release/w375-status-more.png)). The More button is 58 × 26 px, which meets WCAG 2.2 target size (24 px) but not the 44 px guidance; consider enlarging it. |
| S2 | **The mussel quarantine is due to end.** CDPH says it continues "through at least October 31". After that date the record needs a human decision, based on the next CDPH release. Never let it expire automatically. | Recheck on 2026-11-01. |
| S3 | **Data download for researchers.** | [Proposal](release/station-csv-export-proposal.md): link to the source CSV first, then add an in-page CSV. |
| S4 | **Public URL, name and contact.** | Decide on a public domain, check the "CoastWatch" name against NOAA CoastWatch (NOAA's program uses the same word), and set up a contact for corrections. |
| S5 | **Privacy note** for the one item kept in browser storage (the first-visit card dismissal) and for any analytics. | The site has no analytics today. |

## 3. VoiceOver test script

Run it on macOS Safari (VO + arrows, rotor) and on an iPhone (VoiceOver gestures). The task passes if it can be done without sight and nothing is announced misleadingly.

1. On any page, the first swipe or Tab reaches "Skip to main content"; activating it moves to the content.
2. The rotor's headings list shows one h1 per page and a sensible h2 order; the landmarks list has no duplicates.
3. "Not verified" is announced with every notice list, and the status line's "More" reports expanded/collapsed.
4. Map: search "Morro", choose Morro Bay, hear its notices first, then the forecast. The map canvas itself is not the only route to any information.
5. Bloom: change the station; the first measured value and its date are announced with their units, and "reported 0 (not quantified)" is read as such.
6. Fisheries: use the section links. Each chart has a text or table alternative that is read in order.
7. The notices drawer opens, focus stays inside it, Escape closes it, and focus returns to the button.
8. Note anything that is silent, duplicated or ambiguous, with page and step.

## 4. Operational state (2026-10-12)

- **Data refresh:** the last 7 scheduled runs succeeded (one cancelled run on 2026-10-11 05:49 was superseded), and health has 0 alerts. VIIRS has had no scenes for 7 days, so the satellite source is marked partial (an upstream satellite gap, shown in the app).
- **Open issues:**
  - #12, official pages changed: closed by B1.
  - #29, C-HARM history caching and the 2-day carry limit.
  - #22, a Research Preview map load event, to fix in a future demo release.
- **Preview:** Vercel project `coastwatch-preview`, behind a login. It has no production deployment, and its source is identical to `main`'s `coastwatch-web`.
- **Upstream dependencies:**
  - NOAA ERDDAP intermittently refused GitHub runners on 2026-10-09/10 (see the [NOAA outreach draft](p3/noaa-outreach-draft.md)).
  - SCCOOS ERDDAP reloads cause brief 404/502 responses, which are retried.
  - The map tiles come from OpenFreeMap, with a bundled coastline fallback and self-hosted label glyphs.

## 5. What a public page must keep saying (public-facing limitations)

These are already on the site, and must stay there in any public release.

- CoastWatch is not an official source. Closures and advisories come only from CDFW, CDPH and OEHHA, and **a missing notice does not mean an area is open or seafood is safe**.
- Notices are transcribed and **not verified** until a person reviews them, and the list may be incomplete or out of date.
- C-HARM is NOAA's experimental model: a **probability** of conditions, not a measurement, not a toxin level in seafood, and not available for many cells near shore.
- Station measurements describe the water sampled at that pier on that day, usually reported days to weeks later. They do not describe nearby waters or seafood. A reported 0 means "not quantified", not "absent".
- Satellite chlorophyll is a bloom proxy, not a toxin measurement.
- Fisheries figures are historical landings, not losses, forecasts or a judgment about any fishery. The species tiers are an editorial grouping that awaits review (B2).
- The research publication concerns chlorophyll forecasting. CoastWatch does not run those models, and the publication does not validate this application.

## 6. Decisions for the owner

1. **Reviewer:** who to approach for B2 (the draft leaves the recipient blank), and whether to send it before or after B1.
2. **Data download (S3):** approve the proposal's step 1 (a source-CSV link) as a small follow-up PR, or wait for the scientific review.
3. **B4 fix:** approve the small masthead change for 320 px.
4. **Public release shape:** a new public Vercel project and domain for the full website, separate from coastwatch-demo, or replacing the demo later. Either way, only after B1–B4.
