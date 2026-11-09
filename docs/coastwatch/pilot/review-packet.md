# Review packet: pilot fisheries status (about 30 minutes)

Claude transcribed these records on 2026-11-08 from the official pages linked below, into `data/curated/fisheries-status/status.yaml`. **All of them are `pending_human_review`**, and the app shows each one as **"Not verified"** until you complete this review. Only you can verify them; no code does it.

## How to review
1. Open `data/curated/fisheries-status/status.yaml` beside this checklist.
2. For each record:
   - open its `source_url` (and `other_sources`) and confirm what is listed under "Check";
   - fix anything wrong in the YAML (a PR or a local edit);
   - tick the box.
3. Once every record you're vouching for is ticked, record the review:
   ```bash
   cd pipeline
   uv run cwp review-fisheries-status --reviewer "Your Name" --confirm                    # every record + RAMP
   uv run cwp review-fisheries-status --reviewer "Your Name" --ids salmon-commercial-2026 --confirm   # only some
   ```
   - The command refuses to run without `--confirm`, or with a missing, blank, or bot reviewer name.
   - It writes `review.json` with your name and a hash of each reviewed record. If a record is edited afterwards, it goes back to pending.
   - Commit `review.json`. The next data run publishes `human_verified` together with your name.
4. The official-notice registry has its own review (`cwp review-official`); this review does not touch it.

Never verify a record you didn't check today. If an official page has changed, edit the record (date, wording, level) before verifying it.

## Checklist

- [ ] **dungeness-commercial-2026-27** (delayed, whale entanglement). Source: [CDFW Whale-Safe Fisheries](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries) and the [Oct 30 Director's Declaration](https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=247860&inline). Check:
  - Zones 1–5 are delayed to at least Jan 1, 2027, citing humpback entanglements;
  - a trap prohibition and a Fleet Advisory are in place;
  - the next assessments are Dec 15–18 and Jan 13–15;
  - the page labels the Oct 30 PDFs "October 30, 2027". This looks like a typo on CDFW's side.
- [ ] **dungeness-recreational-2026-27** (restricted). Sources: the same page, the [Oct 30 news release](https://wildlife.ca.gov/News/Archive/recreational-dungeness-crab-fishing-begins-in-most-areas-nov-7-with-trap-restrictions-commercial-season-delayed) and [CDPH SN26-023](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-023.aspx). Check:
  - the season opened Nov 7;
  - traps are prohibited in Zones 1–5 until at least Jan 1;
  - hoop nets and snares are allowed (hoop nets serviced every 2 h);
  - the bounds of the north-coast domoic-acid closure. **The sources disagree**: the Whale-Safe page says Reading Rock SMPA 41°17.6′ N, while CDPH and the release say Cape Mendocino 40°10′ N. In both cases the southern edge is Ten Mile SMR, 39°33.3′ N.
- [ ] **anchovy-monterey-bay-2026-09** (restricted, domoic acid). Sources: the [CDFW Sep 11 release](https://wildlife.ca.gov/News/Archive/commercial-and-recreational-take-restrictions-in-place-for-northern-anchovy-in-monterey-bay-due-to-public-health-hazard) and [CDPH SN26-019](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-019.aspx). Check:
  - the restriction covers northern anchovy only, Pigeon Point (37°11.000′ N) to Point Lobos (36°31.461′ N);
  - bait use is allowed;
  - no lift has been announced;
  - it applies to Moss Landing, Monterey and Santa Cruz, and not to Princeton.
- [ ] **mussel-quarantine-2026** (held at **watch**). Source: [CDPH SN26-022](https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-022.aspx) (Oct 28): the quarantine ends at midnight Oct 31, 2026. **Decision for you:** keep it at `watch`, or change the level to `no_restriction_on_record` with a detail that says the quarantine ended Oct 31. I didn't make this change myself, because nothing is expired automatically.
- [ ] **salmon-commercial-2026** (closed, season). Source: [CDFW Ocean Salmon](https://wildlife.ca.gov/Fishing/Ocean/Regulations/Salmon). Check that the page says "The 2026 commercial ocean salmon fishery has concluded." Note that the plan's line "commercial salmon has reopened, with trip limits" is outdated.
- [ ] **rock-crab-central-coast-2026** (no restriction on record). Source: [CDFW Health Advisories](https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories). Check that:
  - the only commercial rock crab closure is Mendocino/Humboldt line to Cape Mendocino;
  - the CDPH viscera advisory covers only the area north of the Mendocino/Sonoma line and the Northern Channel Islands.
- [ ] **ramp (2026-27)**. Sources: the Whale-Safe page and [BIOS ds3120](https://filelib.wildlife.ca.gov/Public/BDB/GIS/BIOS/metadata/DS3120.html) (zone bounds per 14 CCR 132.8(a)(7)). Check:
  - zone bounds: Zone 3 runs from the Sonoma/Mendocino line to Pigeon Point and covers Princeton; Zone 4 runs from Pigeon Point to Lopez Point and covers Moss Landing, Monterey and Santa Cruz;
  - the action text per zone;
  - last assessment Oct 30; next Dec 15.
  - Zone 7 (the leatherback foraging area, Point Arena to Point Pinos) isn't modelled: the current page shows no action for it.

## Not transcribed (no official status found on 2026-11-08)
- Groundfish, market squid and spiny lobster. The CDFW Inseason page lists no current action, and Health Advisories lists no toxin closure for lobster.
- RAMP zone polygons. The ds3120 feature service exists, but its metadata states no use limitations, so only the latitude bounds are used.

## Biotoxin tests (not part of this review)
`v1/biotoxin/index.json` is parsed automatically from CDPH's PDFs. It reports measured results and makes no status claims, so it isn't reviewed here. Spot-check one row against [CrabDAResults.pdf](https://www.cdph.ca.gov/Programs/CEH/DFDCS/CDPH%20Document%20Library/FDB/FoodSafetyProgram/DomoicAcid/CrabDAResults.pdf) if you like, for example BMA G Monterey Bay Dungeness, 9/28/2026, average 5.5 ppm.
