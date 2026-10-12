# 26 — Release readiness

**Status (2026-10-12):** M8 is merged (`main` 69d0d6e8); the 320 px fix and the source CSV link are in PR #32. The full website exists only as a private Vercel preview behind a login. The public Research Preview (coastwatch-demo) is unchanged. **The full website is not ready for a public release** until every blocker in §1 is closed.

Working materials:
- [Official-notice review checklist](release/notice-review-2026-10-12.md) (blocker B1)
- [Independent scientific review: request draft and reviewer checklist](release/scientific-review-request.md) (blocker B2)
- [Proposal: station observation CSV export](release/station-csv-export-proposal.md) (not a blocker; option 1 is in PR #32)

## 1. Blockers (all must be closed before any public release)

| # | Blocker | Who | How it is closed |
|---|---|---|---|
| B1 | **Official notices have never been checked by a person.** The 10 records were transcribed by an AI assistant; issue #12 has stayed open since 2026-10-10. Open questions: (1) the CDFW page still links CDPH's *lifted* sardine and anchovy advisory (SN26-013, lifted by SN26-016), and the current advisory (SN26-019) names anchovies only; (2) CDPH's Humboldt razor clam advisory (SN24-003, still in effect per SN26-020) is missing from the registry; (3) the mussel quarantine runs "through at least October 31". | Owner | Work through the [checklist](release/notice-review-2026-10-12.md), edit the registry where needed, then run `cwp review-official --reviewer "<your name>" --confirm`. Close #12 only after the next watcher run matches. |
| B2 | **No independent HAB scientist has reviewed the application.** This covers how C-HARM is described, how station observations are read, the fisheries species tiers and the scientific wording. | External reviewer | Send the [request](release/scientific-review-request.md); record the findings and the changes made. The About page's "Review status" is updated only after the review has happened. |
| B3 | **No real screen-reader test.** axe and the accessibility tree are clean (0 violations in 16 states, doc 25 §5), but no person has used the site with VoiceOver. | Owner or tester | Do the [VoiceOver pass](#3-voiceover-test-script) on macOS Safari and iOS Safari, and fix anything that blocks a task. |
| B4 | **At 320 px wide the masthead overflowed by 28 px.** The page scrolled sideways and the notices pill was clipped at the right edge ([before](release/w320-map.png)). WCAG 1.4.10 (Reflow) requires 320 px. | Claude | **Fixed in PR #32** (open; closes when merged). Below 360 px the pill drops its icon and keeps the count and "Not verified". Tested at 320, 360, 375, 390, 1024 and 1440 px on five pages; axe finds 0 violations. |

## 2. Should be done before a public release (not blocking)

| # | Item | Note |
|---|---|---|
| S1 | **The status line wraps at 375 px.** | Checked 2026-10-12, **acceptable**: it wraps to 2 lines (46 px instead of 35 px). "Notices not verified. Check CDFW and CDPH." and the More button stay fully visible. More opens the full reason and the link to the list, and nothing is clipped. No page scrolls sideways at 390, 375 or 360 px ([375 px](release/w375-map.png), [More open](release/w375-status-more.png)). The More button is 58 × 26 px, which meets WCAG 2.2 target size (24 px) but not the 44 px guidance; consider enlarging it. |
| S2 | **The mussel quarantine is due to end.** CDPH says it continues "through at least October 31". After that date the record needs a human decision, based on the next CDPH release. Never let it expire automatically. | Recheck on 2026-11-01. |
| S3 | **Data download for researchers.** | Option 1 (a source CSV link on Bloom) is **in PR #32**. Option 2 (an in-page CSV with qualifiers) waits for the scientific review; see the [proposal](release/station-csv-export-proposal.md). |
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

## 6. Public deployment plan (prepared, not executed)

The full website gets its **own public Vercel project**. The Research Preview (`coastwatch-demo`) and its deployments stay exactly as they are, and the private review project (`coastwatch-preview`) stays private and keeps being used for previews.

| Step | What | Gate |
|---|---|---|
| 1 | Create a new Vercel project, for example `coastwatch-site`. Use framework Next.js and root `coastwatch-web`. **Disconnect Git auto-deploys**, as for the demo. Set `CW_DATA_BASE_URL=https://yashnil.github.io/habs-forecast/v1` for production and preview. Keep deployment protection on until launch. | Owner approves creating the project |
| 2 | Choose a domain and name. The word "CoastWatch" is also NOAA's program name, so consider a distinct public name or subtitle. Set up a contact address for corrections. | Owner decision |
| 3 | Add response headers in `next.config.ts`: a Content-Security-Policy allowing the map tile host (OpenFreeMap), ERDDAP links and the data host, plus `Referrer-Policy` and `X-Content-Type-Options`. The app sets none today. Add `robots` metadata that allows indexing only from launch. | Small PR, with tests |
| 4 | **Release candidate:** tag the reviewed commit `site-release/YYYY-MM-DD-rc`. Deploy it to `coastwatch-preview` (private) from a clean `git archive` export. Run the e2e suite, axe and `cwp check-published` against it. The owner walks the six journeys. | B1–B4 closed; CI green |
| 5 | **Launch:** deploy the same tag to the new project with `vercel deploy --prod`, then remove deployment protection. Record the deployment id and the tag in this file. | Owner's explicit go, given at that time |
| 6 | **Rollback:** promote the previous production deployment (`vercel promote <dpl>` or `vercel rollback`). Release tags are never moved or deleted (as with `demo-release/*`). | — |
| 7 | **After launch:** the existing data-health alerts and the official-page watcher keep running. Add an uptime check on the home page and `manifest.json`. Recheck the notices after each watcher alert, and recheck the mussel quarantine each autumn. | — |

Nothing in this plan changes the data pipeline or the `coastwatch-data` branch. The site reads the same published data as the private preview.

## 7. Decisions for the owner

1. **Reviewer:** choose from the shortlist (kept outside this public repository because it names individuals) and decide whether to contact them before or after B1.
2. **Notices:** whether to add the CDPH Humboldt razor clam advisory (SN24-003) as a record; and whether to tell CDFW that its page links the lifted sardine and anchovy advisory.
3. **PR #32:** merge after review (B4 and the source CSV link).
4. **Public release:** the project and domain (§6 steps 1–2), and when, only after B1–B4 are closed.
