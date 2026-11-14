# A. How commercial fishermen on the California coast prepare a trip

Research track A for the Neritic v12 redesign. Desk research, 2026-11-10. **Nobody was interviewed or contacted. No fisherman has used or commented on Neritic.** The dockside interview script (`../../pilot/pitch/interview-script.md`) has not been run.

**How to read the tags**
- A plain link means the page or PDF was read.
- **(search)** means only a search-engine summary was seen; the page could not be opened or was not opened. Treat it as a lead.
- **(repo)** means a finding carried from earlier repo research with its own source ID ([01-literature](../../fishermen-first/01-literature.md), [02-existing-products](../../fishermen-first/02-existing-products.md), [stakeholders](../../research/stakeholders-and-interviews.md)). Those were not re-read today.
- **(inference)** means my reasoning. No source says it.
- **(gap)** means I looked and found nothing.

**The biggest limits**
1. I found **no study, survey or first-person account of what a California commercial fisherman checks before a trip, or in what order.** The closest evidence is an Oregon interview study (11 fishermen, 4 spouses, 1 industry representative, mostly Newport).
2. I found **no evidence of which weather or chart apps California commercial boats use.** Product pages say what apps do, not who on the docks uses them.
3. I found **no study of phones, gloves, or sunlight on West Coast boats.**
4. Evidence on **frustrations** is much stronger for season status and rules than for weather.

Sections 2, 3 and 8 are therefore partly reconstruction. Section 11 lists what dockside interviews must settle.

---

## 1. Who they are

### Commercial vs recreational: the job differs

| | Commercial | Recreational | Source |
|---|---|---|---|
| Same crab, this week | Delayed in Zones 1–5 until at least Jan 1, 2027 because of humpback entanglements | Season opened Nov 7 with traps banned; hoop nets and snares allowed | [CDFW bulletin](https://content.govdelivery.com/accounts/CNRA/bulletins/4312787) |
| Domoic acid closure in force now | Status follows from the delay above | Take and possession prohibited in a northern stretch of state waters, stated two ways (see note) | [CDFW bulletin](https://content.govdelivery.com/accounts/CNRA/bulletins/4312787), [Whale Safe page](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries) |
| What is at stake | Income, with gear soaking at sea: "I just lost my crab gear because we had a 28-foot sea that I didn't foresee coming" | Not found as a livelihood; a day and a meal (inference) | [Kuonen et al. 2019](https://repository.library.noaa.gov/view/noaa/43899) |
| Who decides they can sell or keep | A licensed buyer, a landing receipt, and the permit | A bag limit and a consumption advisory (inference) | [CDFW landings](https://wildlife.ca.gov/Fishing/Commercial/Landing-Resources) |
| Compliance load | Permit, electronic tracking, twice-monthly reports, fish tickets, logbooks in many fisheries (section 4) | License, report cards where applicable (inference) | [CDFW MFSU](https://wildlife.ca.gov/Conservation/Marine/MFSU) |
| Can they skip a bad day | Fixed costs continue; 56% of West Coast owners "have fished when profits might not cover vessel expenses in order to employ crew" | Yes (inference) | L15 (repo) |

**Note on the current recreational closure.** The CDFW bulletin says Cape Mendocino (40° 10' N) to Ten Mile SMR. The Whale Safe page, updated Oct 31, says Reading Rock SMPAs to Ten Mile SMR. These differ. Check CDFW before any copy names the boundary. A product that shows one without its date will be wrong for someone.

The recreational and commercial questions should not share one headline. A commercial crabber cannot use "season open" (inference).

### The fleet in numbers

- **2,857 registered commercial vessels** in 2024 (575 of which were registered CPFVs) and **5,366 commercial licenses**. [CDFW 2024 By The Numbers](https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=229396&inline=)
- Landings by port area, 2024, value: Santa Barbara $65.0M, Los Angeles $37.5M, San Francisco $26.5M, Eureka $20.0M, San Diego $12.1M, Bodega Bay $10.9M, Monterey $7.9M, Fort Bragg $5.4M, Morro Bay $3.7M. Same report (transcribed in `coastwatch-web/src/content/landings.ts`).
- **Design consequence.** The pilot's four deep ports are all Monterey Bay and San Mateo coast (`PILOT_PORTS`, `contracts.ts`). By 2024 value, Monterey is seventh of nine port areas. Today's curated status has six records: Dungeness commercial and recreational, Monterey anchovy, mussels, commercial salmon, Central Coast rock crab (`data/curated/fisheries-status/status.yaml`). None covers squid or lobster, the two fisheries that likely dominate the top two port areas (inference).
- Most boats are small. The literature's dividing line is 40 ft. Size predicts **mobility**, not whether a boat stops fishing; large boats can work two port areas, small boats generally cannot (Jardine 2020, Fisher 2021, repo).

### Main fisheries

| Fishery | 2024 value | Where and who | Season and gates | Note |
|---|---|---|---|---|
| Market squid | $67.9M, largest ([CDFW](https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=229396&inline=)); OPC says $78M for 2024 (search), an unresolved difference | Monterey Bay and Channel Islands (OPC, search). Seine vessels and light boats, permit-limited | Apr 1–Mar 31; 118,000 short ton limit; closed noon Friday to noon Sunday ([14 CCR 149](https://law.cornell.edu/regulations/california/14-CCR-149)) | Light boats attract squid. Season close is announced on VHF 16 between 10 p.m. and midnight, and that is official notice. Same source. |
| Dungeness crab | $49.7M | Northern ports most dependent; ~26% of fishery revenue, >25% of vessels (Fisher, repo). Seven pot tiers, 175–500 ([FGC 8276.5](https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=FGC&sectionNum=8276.5), repo preset) | Nominal Nov 15 (Districts 6–9 Dec 1) to Jun 30/Jul 15 ([calendar](https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=156448&inline)). Real gates: RAMP, domoic acid, meat quality, fair-start, price | Landings concentrated in the first six weeks (L5, repo), so delays remove the valuable weeks |
| Spiny lobster | $20.9M | Southern districts 18, 19, 20A and part of 20 | Oct 7, 2026–Mar 17, 2027 ([calendar](https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=156448&inline)); daily log DFW 122 ([MFSU](https://wildlife.ca.gov/Conservation/Marine/MFSU)) | In season now. A choice in the account's fishery list, but no curated status record (`status.yaml`) |
| Red sea urchin | $7.5M | Dive boats. Northern fishery hit by kelp loss (search) | Open and closed days in the Digest (calendar); dive log DFW 120.7 | Details not retrieved (gap) |
| California halibut | $5.8M | Hook-and-line, trawl, gill net | Hook-and-line all year; trawl grounds Jun 16–Mar 14 ([calendar](https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=156448&inline)) | Pacific halibut commercial in Area 2A not found (gap) |
| Sablefish and groundfish | Sablefish $4.7M | Fixed gear and trawl. Limited-entry and open-access sectors. The fallback for northern crab boats in 2015–16 (L6, repo). Ports not broken out (gap) | Two-month cumulative limits, depth closures, frequent in-season changes; sablefish primary fishery Apr 1–Dec 31 north of 36° N (search). Nearshore open all year, quillback may not be taken ([calendar](https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=156448&inline)) | Federal VMS and declarations apply (section 4) |
| Salmon | Not in the 2024 top ten; fishery closed that year | Fort Bragg, San Francisco, Monterey ~85% of 2019–22 Chinook (search) | 2026: first commercial opening since 2022. Closed Oregon border to Point Arena. San Francisco area: four multi-day openers in May, three in August. Monterey area: multi-day openers May–August. All subject to the statewide limit ([PFMC](https://www.pcouncil.org/documents/2026/04/pacific-fishery-management-council-adopts-2026-west-coast-ocean-salmon-seasons-updated-april-12-2026.pdf/)). First year of vessel trip limits ([CDFW](https://wildlife.ca.gov/News/Archive/ocean-salmon-fishing-comeback-continues)) | Concluded for 2026 (search). Short windows make notice timing matter |
| Anchovy, sardine (CPS) | Not checked | Monterey Bay | Anchovy and sardine restricted Pigeon Point to Point Lobos since Sep 11, 2026, for domoic acid (`pilot/plan.md`, not re-verified today) | A live example of toxin affecting a non-crab fishery |
| Spot prawn | Not checked | Traps | South of Pt Arguello Feb 1–Oct 31, ≥50 fathoms; north Aug 1–Apr 30 ([calendar](https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=156448&inline)) | |
| Albacore | Not in CDFW's top ten | Troll, offshore. Vendor reports Starlink on albacore boats within 200 miles (search) | Not on the state calendar; federally managed HMS (inference). Details (gap) | Fishing off the edge of what Neritic shows, by distance |

---

## 2. What they check before leaving

### What is actually documented

**From the Oregon interviews** ([Kuonen, Conway & Strub 2019](https://repository.library.noaa.gov/view/noaa/43899), methods, results and discussion read; n = 16, of whom 11 fishermen; authors say results reflect Oregon's context):
- Forecasts are "used for planning when to go, where to go, and how long to stay out", and the last two "may evolve once a fisherman is at sea".
- They **compare**: forecasts against personal observation, real-time data and other forecasts. "They compare multiple forecasts of the same variable to see how well they agree."
- **Real-time data earns the trust.** "The buoys offshore show us what it's actually doing: not just the prediction but what it's actually doing."
- They want the **longest possible view** despite knowing skill decays: "I like as many days as possible... it's stupid because it changes so drastically... It makes me feel better for some reason."
- **Vessel size and captain experience** are "the most important factors" in risk orientation. Vessels under 60 ft have much lower tolerances and "rely more heavily on forecasts".
- A simple page with "near-term forecasts and real-time information for multiple ocean variables for a location" is what the authors say fits. Locations called out: "bar crossings and fishing grounds".
- The marginal day is the hard day. Fishermen "tend to be more concerned with marginal conditions when the decision is less clear". "It's defining that line between what's unsafe and what's just uncomfortable."
- Companion paper: fishermen combine sources non-linearly, with no single trusted source ([MTS Journal](https://doi.org/10.4031/mtsj.53.1.1), search).

**From the team that built a forecast tool with those fishermen** ([Strub, Seacast slides](https://www.star.nesdis.noaa.gov/data/star_docs/meetings/2019OSOS/Thursday/4_StrubT.pdf)):
- "The presence of many choices is initially a deterrent to use. Their activity at sea is hectic, stressful and dangerous."
- "They prefer to look at forecast fields **on land**, under calmer conditions", which requires forecasts reaching past the first couple of days.
- They "**do not want fields depicting uncertainty or expected error**"; they compare against experience and other data instead.
- Units are knots, fathoms, degrees F. A user-driven product needs these by default.
- Fishermen asked for things the scientists "could not have guessed": current direction for setting gear, strong currents dragging marker buoys under, and **low-salinity water killing crab held aboard or in bays while waiting to deliver**. "The models don't include bays."
- The full data explorer was "too complex", so a simplified page was built and iterated with a small group of one port's fishermen. Seacast later moved to NANOOS (search).

**From Newfoundland** (Finnis et al. 2019, abstract via [PubMed](https://www.pubmed.gov/31293225/), search): decisions turn on "nuanced and contextual" reading of a few hazards, chiefly wind and to a lesser extent waves. Harvesters care more about usefulness than accuracy.

**The one documented order of steps on this coast** is NWS Eureka's bar guidance ([North Coast hazards guide](https://www.weather.gov/media/eka/northcoasthazards.pdf)). It is written for boaters generally, so it is advice, not observed behaviour:
1. Coastal Waters Forecast.
2. Bar forecast for departure time; otherwise check the tide is coming in and waves are under five to six feet.
3. Bar forecast **for the return**: "it can be more dangerous coming into the bay".
4. "Right before you leave", current conditions from the Coast Guard (707-443-2212 or VHF 22A) "or other boaters who have recently crossed", cross-checked against the bar cam. A second number, 707-443-2213, appears on an older NWS flyer (search); the current one needs confirming. "Checking... before leaving the dock is better than 'going to have a look'."

### The bar is a regulated decision point

[33 CFR 165.1196](https://www.law.cornell.edu/cfr/text/33/165.1196) creates regulated areas at **Humboldt Bay, Noyo River, Crescent City and Morro Bay**. A restriction bars recreational and uninspected passenger vessels; a closure bars everyone. Both are announced on VHF 16 and 22A. Commercial fishing vessels may cross under a restriction but must have lifejackets or immersion suits on anyone in unenclosed areas and call the Coast Guard on channel 16 before crossing (name, position, people aboard, destination). Unsafe waves follow L/10 + F = W, where L is length and F is freeboard. Local go/no-go plans exist.

**Neritic currently has nothing about bars** (a search of `coastwatch-web/src` found no bar forecast or report). Neritic should **not** give a bar go/no-go, because the rule depends on the vessel and on-scene judgment. `pilot/plan.md` §3D already lists bar crossing among the questions Ask must refuse. A link-and-time line ("Humboldt bar forecast, NWS, issued..."; "Coast Guard report, VHF 22A") costs little and sits in the highest-consequence step of the morning (inference).

### A reconstructed preparation sequence (inference, not observed)

| When | Likely check | Sources that support it |
|---|---|---|
| Days out, on land | NWS zone forecast, multi-day models, season and rule status, buyer contact | Kuonen; Strub "on land" |
| Night before | Forecast again, tide for the bar or launch, crew and ice arrangements | Inference; squid notices arrive 10 p.m.–midnight by rule |
| Early morning, dock or truck | Buoys and observed wind, latest forecast update, bar report, other boats | NWS guide; Kuonen |
| Underway | VHF weather and traffic, radio talk, own instruments | 33 CFR; Kuonen; radio use (L25, repo) |

### Tools

| Tool | What it does for this job | Price (checked) | California commercial use |
|---|---|---|---|
| NWS zone forecast, NOAA Weather Radio | Official wind and seas by zone | Free | Rated "solved well" in [doc 02](../../fishermen-first/02-existing-products.md) |
| NDBC buoys, CDIP | Observations. The NDBC mobile page is 2.6 KB | Free | As above |
| CO-OPS tides | Tide predictions | Free | As above |
| PredictWind | Model comparison, departure planning, observations | [$0, $29, $249, $499 a year](https://www.predictwind.com/pricing) | (gap) |
| Windy | Free model maps, comparison | Not checked | (gap) |
| Buoyweather | Point sea-state forecasts (search) | Not checked | (gap) |
| FishTrack | SST, chlorophyll, offline imagery | [$14.99/month, $95.99/year](https://apps.apple.com/app/id952022603) | National Fisherman covered it ([article](https://www.nationalfisherman.com/viewpoints/boats-gear/reeling-data-fishtrack-app), blocked) |
| RipCharts | SST, chlorophyll | ~$170 a region ([third-party page](https://adapty.io/paywall-library/ripcharts/), unverified) | (gap) |
| TimeZero Professional | Plotter software with seabed mapping and trawl modules. [Vendor](https://mytimezero.com/tz-professional/solution-for-commercial-fishing/) | ~$1,500; Ocean-O add-on ~$800 (retailer, search) | (gap) |
| Olex | Seabed maps from the vessel's own sounder, shared between users ([Olex](https://olex.no/index/fishing_en.html)) | Not published | (gap) |
| Deep Zone | **I could not identify a product by this name.** The source of this name should be asked | | |
| VHF, other boats, buyers | Immediate, local | Free | Kuonen; bar rule; radio use (L25, repo) |

The point for design: **nothing in Neritic's data competes with the plotter** on navigation or fish-finding. Those are different jobs (inference).

---

## 3. How they decide a trip is worth it

There is little published detail on the go/no-go calculation. Kuonen reports only that the need to remain profitable sometimes requires working in hazardous conditions, and fishermen stay at sea "as long as possible until the weather or catch limit causes them to return".

| Input | What is documented | Who already knows | Can Neritic add anything? |
|---|---|---|---|
| Ex-vessel price | **Set by annual negotiation.** The Humboldt Fishermen's Marketing Association and Pacific Choice Seafood reached $4.75/lb on Jan 30, 2026, the opening morning; last year's $5.75, first offer $4.50. These talks "usually set the price for the entire West Coast" and can bring strikes ([Lost Coast Outpost](https://lostcoastoutpost.com/2026/jan/30/commercial-crabbers-reach-deal-price-will-start-se)). 2021 settled at $2.75 after weeks of haggling (search) | The association and buyer, before anyone | No. Past statewide averages only (CDFW, already labelled in the app). |
| Fuel | "A major component of variable costs" (Dewees, repo). EFIN puts fuel at 10–15% of variable cost (search; age unclear). California retail diesel was $8.24/gal the week of Oct 19, 2026, against $4.90 a year earlier (EIA, in `referenceValues.ts`; road diesel, not marine) | The skipper and fuel dock | Weak. A reference, and only that. |
| Crew | Crew are commonly paid a share; NOAA survey forms ask which trip expenses come off first ([EFIN](https://www.psmfc.org/efin/surveys/woc_surv.pdf), search). Fixed-gear crew 28–31% after deductions in 2008 (NWFSC-121, repo preset). Crew leave during delays: "They're taking outside jobs... We don't know when we're going" ([Glickman et al. 2025](https://doi.org/10.1038/s44183-025-00139-0)) | The skipper | **Yes, indirectly.** Earlier notice of opening dates is what lets them hold crew. |
| Ice | A real port constraint. Noyo's icehouse is "very unreliable", and "many of the big boats sometimes have to go all the way down to Bodega Bay" ([Mendocino Voice](https://mendovoice.com/2025/10/new-noyo-harbor-icehouse-poised-to-support-the-north-coast-fishing-industry/)). Spud Point is the only provider of its kind between San Francisco and Fort Bragg; Crescent City and San Francisco have had similar problems; Eureka about $350 a ton against $140 elsewhere (all search) | Harbor office, dock | No feed exists. Leave out. |
| Bait | (gap) | | |
| Quota and limits | Squid seasonal limit; two-month groundfish cumulative limits; salmon vessel trip limits and a statewide limit; urchin open days; pot tiers | The skipper knows their catch. In-season changes come from agencies | **Partly.** Show the *rule and its change date*. Never a personal tally. |
| Buyer order | Whether a buyer will take product on a given day. I could not confirm boat-by-boat limits (gap). Only a few companies can eviscerate crab in a closure: "There's just a few big companies that actually can eviscerate the crab" (L18, repo) | Buyer and skipper, by phone (inference) | No. |
| Days at sea | Weather window or limit | The skipper | The multi-day forecast. |

**Existing code.** `v4/coast/economics/` already has a Trip Economics worksheet (catch lines, costs, crew share, EIA diesel, CDFW past-average prices, saved scenarios on this device). Nothing shows a fisherman asked for it. Its inputs are ones fishermen already hold. Skippers have spreadsheets or a settlement sheet from buyers (inference). It should not sit in the principal journey before interviews test it.

---

## 4. Rules and what each one requires

| Measure | Who decides, what it is | How it reaches people | What this means for them | Neritic today |
|---|---|---|---|---|
| RAMP season delay and restrictions | CDFW Director, by Zone 1–6: fleet advisory, 15–50% gear reductions, 30-fathom limit, delay, closure, pop-up gear authorization (L11, repo). Now: Zones 1–5 delayed to at least Jan 1, 2027 ([CDFW](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries)). Next assessments **Dec 15–18 and Jan 13–15, subject to change** (same page; confirmed today) | PDFs on Whale Safe, GovDelivery list, news | Decides crew, gear, bait, buyer plans. Recent declarations gave 13–17 days before effect (L11, repo) | Transcribed YAML, pending human review |
| Domoic acid in crab | CDPH tests, OEHHA assesses, CDFW acts. FDA action levels are 20 ppm meat and 30 ppm viscera. Pre-season samples from representative sites (CDPH-CDFW documents, search). Reopening is reported as two consecutive sets of six crabs at least seven days apart (SF Chronicle 2015, unverified, repo) | Rolling CDPH PDF, CDFW Health Advisories, phone lines (831) 649-2883 and (800) 553-4133 ([CDFW bulletin](https://content.govdelivery.com/accounts/CNRA/bulletins/4312787)) | Decides whether crab can be sold whole. In 2025–26 the BMA B delay was announced 7 days before the planned opener; clearance Jan 23, open Jan 30, 64-hour gear-setting (repo; search). | CDPH parser in sidecars |
| Meat quality | Pre-season quality testing can delay Districts 6–9 ([calendar footnote](https://nrm.dfg.ca.gov/FileHandler.ashx?DocumentID=156448&inline)) | CDFW, PSMFC | A third independent gate on the opener | Not separately shown (inference from the facts list; confirm) |
| Fair start | Vessels that fished elsewhere are barred from a later-opening area for a period (FGC 8279.1, L7, repo); in 2026, until March 1 for BMA B (repo) | Statute, CDFW | Changes whether moving boats is worth it. Only matters to mobile, mostly large vessels | Not shown (inference) |
| Bar restriction or closure | Coast Guard Captain of the Port ([33 CFR 165.1196](https://www.law.cornell.edu/cfr/text/33/165.1196)) | VHF 16/22A, warning lights, sometimes cameras | Immediate | Nothing |
| Season notices | PFMC and NMFS (salmon), CDFW (others). Squid: VHF 16, 10 p.m.–midnight, official ([14 CCR 149](https://law.cornell.edu/regulations/california/14-CCR-149)). NMFS salmon hotline (800) 662-9825, CDFW (707) 576-3429 ([CDFW](https://wildlife.ca.gov/News/Archive/ocean-salmon-fishing-comeback-continues)) | Radio, hotline, web | Hours of notice, in some fisheries | Partial |
| Crab electronic monitoring | Every Dungeness vessel needs a device reporting location at least once a minute during trips, from 2025–26 ([CDFW](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries)). Cellular or satellite; reportedly a return-to-port rule if transmission fails (CDFW presentation, search). Lime units via Archipelago ([PCFFA](https://www.pcffa.org/news/electronic-monitoring-information)) | CDFW | The boat is **already tracked by a state system**, with data available to CDFW (search). A second tracker is the last thing anyone needs (inference). | None, correctly |
| Twice-monthly activity report | Dungeness vessels report permit, zone, depth range, traps, by the 1st and 16th (14 CCR 132.8, [MFSU](https://wildlife.ca.gov/Conservation/Marine/MFSU)) | Online CDFW form | Paperwork already exists | None, correctly |
| Landing receipts, E-Tix | Required electronically since Jul 1, 2019, through licensed fish businesses or a Fisherman's Retail License. Three business days; 24 hours for bluefin, sablefish, trawl groundfish ([CDFW](https://wildlife.ca.gov/Conservation/Marine/MFSU)). Copy kept four years (search) | Buyer submits | The **official sale record**. Not something Neritic should ask for | None, correctly |
| Logbooks | Squid (DFW 149A/B), lobster, gill net, shrimp trawl, urchin and other dive, some trap. Federal groundfish trawl and HMS with NOAA. Salmon troll logbooks (gap) ([MFSU](https://wildlife.ca.gov/Conservation/Marine/MFSU)) | Paper or NOAA systems | More reporting duties | None, correctly |
| VMS | Federal, Pacific Coast groundfish, with trip declarations ([NOAA](https://www.fisheries.noaa.gov/west-coast/resources-fishing/vessel-monitoring-system-west-coast), 50 CFR 660.14) | Federal | Already reports position | None, correctly |
| Permits | Mostly limited access (L5, repo). A CDFW license app exists (repo) | CDFW | Not information they need from Neritic | None, correctly |

**Also check.** Governor Newsom signed SB 1393 (Dungeness crab fishery changes) on Sep 27, 2026 (search; headline only). It may change the rules above. Someone should read it before the redesign finalizes any crab status copy.

---

## 5. What is manually gathered, and what they complain about

**Hard evidence of frustration (California fishermen's own voices)**
- **Uncertainty about when, more than where:** "We don't know when we're going" ([Glickman](https://doi.org/10.1038/s44183-025-00139-0)); "You used to be able to depend on the opening of crab season like clockwork... It's really hard to know what to do" (L35, repo); "But they don't give us a 30-day notice, right?" (L18, repo).
- **Scattered notices.** Pomeroy & Culver, 20 fishermen and handlers: advisories are "posted on various websites, sent out to those who sign up" and fishers need a "timely notification system — accessible, easily understood" (L18, [paper](https://www.frontiersin.org/journals/sustainable-food-systems/articles/10.3389/fsufs.2026.1881090/full)).
- **Process and distrust.** "Decision-making has not been transparent" (L18). "It all boils back to the trust side of it. When you have zero trust with the people you're working with, you just kind of throw up your hands" ([Glickman](https://doi.org/10.1038/s44183-025-00139-0), Fort Bragg). California fishing respondents rated management effectiveness lowest of any group (S3, repo). Fishermen trust colleagues, then Sea Grant and county staff (S3, repo).
- **Contrast with weather.** Forecasters are seen as "doing the best they can" (Kuonen). I found **no complaint that checking weather across services is a burden**. Calling it a burden is our own assumption.

**The multi-hop work, shown by desk tests in [doc 02](../../fishermen-first/02-existing-products.md).** "Will the crab opener happen for my zone?" takes at least eight hops: Whale Safe, Crabs page, In-Season, Health Advisories, CDPH PDF, PSMFC meat quality, Oregon closures, association or phone. Three independent gates, three publishers, three formats. Groundfish depth limits take 4–6. Weather takes 3–4 and is "mostly solved".

**Fishermen as a source.** They want the reverse channel too: "They're making decisions outside of the field and we're seeing it all in real-time. We could be their support, but our voices are not being heard" ([Glickman](https://doi.org/10.1038/s44183-025-00139-0)). A TNC and association programme already collects fishermen's whale surveys (repo). A feedback channel is plausible but would need a plan for who receives it, and a false-promise risk if nobody at an agency does (inference).

---

## 6. Where HAB and toxin information adds genuinely new value

**How 2015–16 hit them (repo, peer-reviewed).** California landings fell to 52% of the five-year average catch and 58% of revenue (L5). Central opened Mar 26, 2016, about 4.5 months late; Eureka May 26 (L7). Estimated crab fleet revenue loss $26.1M, over half of it from other fisheries, and ex-vessel prices fell 22–23% (L5). 56.4% of northern and 72.5% of central crab vessels stopped all fishing in the early season (L6). "Fishermen were scrambling" (L18). Relief reached participants in late 2019 (L18). Closures can arrive abruptly: a CDPH warning in early June, closure in early November, and a rock crab closure that was "particularly disruptive" (L18).

**What each science product is, and who reads it**

| Product | Gives | Who reads it (evidence) |
|---|---|---|
| CDPH crab DA results | Per-site, per-crab ppm in a rolling PDF | Fishery participants and buyers likely; no study (gap). |
| CDPH and CDFW advisories | Regulatory state | Fishing-industry respondents named CDFW the second most common source; local TV was most common among Californians (S3, repo) |
| HABMAP pier data | Cell counts, particulate DA, weeks to months after sampling for SoCal piers ([sources.md](../../trip-briefing/sources.md)) | Researchers (inference) |
| C-HARM | Model probability, 3 days | No user evaluation found (gap, S-doc). One reported comment, "as Dungeness crab fishermen, we are following these models daily", from a portal feedback item; could not trace (search). Natural resource managers, end users and NOAA partners gave feedback by form ([C-HARM presentation, EPA archive](https://archive.epa.gov/epa/sites/production/files/2017-05/documents/california-harmful-algae-risk-r9.pdf)) |
| CA HAB Bulletin | Experimental, every 2–3 months, latest listed May–June 2026 as of Oct 12 (S14, repo) | Science and management audience (inference) |

**New value, in order of defensibility (inference from the above)**
1. **Crab test values by BMA, per sample set, with date**, next to the threshold, out of the PDF. Nobody else presents these in plain form on a phone (doc 02).
2. **One place for the gates on the next opening**: whale assessment, domoic acid, meat quality, fair start, with the next known date for each and the source. This is exactly the "eight hops", and the chief California complaint is about when, not whether.
3. **Early warning for planning** before the opener: gear, crew, buyer and travel. Dungeness is tested about a month before the opener (`pilot/plan.md` §1.7, repo).
4. **Other fisheries touched by toxin**: anchovy and sardine in Monterey Bay, rock crab, bivalves. Rock crab "had never before been closed" until 2015 (L18). Whether domoic acid affects squid sales is not established (gap).
5. **Recreational crabbers and harvesters** eating their catch; CDPH advice is health-critical (repo). This group may gain more than commercial crews do.

**What this is not**
- A toxin forecast for a trip. C-HARM is a model of bloom likelihood; closures are decided on tests. "C-HARM is not wired into decisions" (S3, repo). It runs three days, was reported about six days stale on Nov 8 (`pilot/plan.md`), and only 12.6% of cells within 1 km of shore have values ([sources.md](../../trip-briefing/sources.md)).
- A reason to treat chlorophyll as toxin. The 8-day outlook is chlorophyll only.
- Reassurance. California has no routine mid-season DA sampling (S7, repo), and Dungeness is retested only when over the limit (`pilot/plan.md` §1.7). A result dated weeks ago needs to say so. Sampling also relies on participants' crab, and there has been "pressure to not provide samples" (L18).

---

## 7. Keep or drop

| Feature | Verdict | Why | Privacy model if kept |
|---|---|---|---|
| **Saved vessel profile** | **Keep, as "my port, my fisheries, length class"**. Drop everything else (name, documentation number, permit numbers, engine, gear count). | Port and fishery order the page. Length class is the one vessel fact with evidence (under 60 ft, or 40 ft for mobility). Pilot already stores `vessel_length_class`. Do not compute bar limits from it: the formula needs freeboard and the decision is the Coast Guard's. | On device by default, usable with no account. Account sync optional, row-level security, one-tap delete. Never shared, never sold. Never ask for a name that identifies a vessel. |
| **Trip log** | **Drop as a feature.** Keep only a one-tap "did this match what you saw?" on forecasts and observations, tied to place and date. | They already face monitors, activity reports, fish tickets and logbooks (section 4). Fishing locations and results are commercially sensitive (inference; industry use of a shared seabed map is a voluntary exception). No evidence of demand. Reporting tools drew "intrusive" complaints elsewhere (iSnapper, search). The existing `TripLogForm` avoids catch and location, which is right, but gives a fisherman nothing back. | If retained for pilot research: explicit opt-in, port and date only, no catch, no position, delete anytime, clear statement of who reads it. Keep it out of the main journey. |
| **Landing record** | **Drop.** | E-Tix is the legal record and buyers submit it. A duplicate invites error and creates a sensitive financial dataset. No user value identified. | None. |
| **Buyer price** | **Drop.** Keep labelled past-average CDFW $/lb only where it already appears, and do not move it forward. | Crab price is set by association and buyer and known first to them. No timely official feed (`pilot/plan.md` §4). A crowd-sourced current price invites wrong numbers, and buyer-seller disputes are live; one undated search result mentioned a fisherman's lawsuit against buyers. Legal exposure of publishing current prices was not researched. | None. |
| **Document upload** | **Drop.** | No evidence of need. Permits and licences have official homes (CDFW license app). Storing ID-linked documents adds real breach risk for a project that says it is a pilot with "Not verified" notices. | None. |

**Keep and strengthen (not asked, but it follows):** saved port and alert when status changes for that port and fishery. It is the stated request (L18). Note it competes with CDFW's own GovDelivery email and text topics (doc 02), so it needs to add what they do not: the single place, the date, and the plain "what changed".

---

## 8. Connectivity, devices, conditions

| Point | Evidence | Design consequence |
|---|---|---|
| Offshore cell | "out of wifi and cellphone range" and radio used to share locations (L25, repo). Crab EM units may be cellular or satellite ([CDFW](https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries)). No California coverage figures found (gap) | The check happens on land or near the harbor. |
| Satellite | Starlink is spreading in the fleet, including West Coast boats, though small operators object to maritime plans and it needs clear sky and power (National Fisherman, Network Innovations, search; much of this is vendor material) | Do not assume it. Do not depend on it. |
| Dockside | No data (gap) | Assume patchy; a phone with some signal near the harbor (inference) |
| Field precedent | A California e-logbook review recommended "local offline storage and transmission once connectivity is restored" ([GreenInfo](https://greeninfo.org/?p=431)) | Saved copy should be easy and say its own age. `offline.ts` already does this. |
| Where they look | Strub: "prefer... on land, under calmer conditions". At sea "hectic, stressful and dangerous" | Design for the truck, the kitchen table, the galley and the wheelhouse first. The deck is a poor place to use this. |
| Sunlight and gloves | **No study found** (gap). General engineering knowledge suggests high contrast and big targets, but "must work on a rocking deck with gloves" is our assumption. | Keep high contrast and large targets, since they cost little. Do not treat "for gloves" as a validated requirement. |
| Age | West Coast owners: mean 58, more than half over 60 (L15, repo). Another source says California captains average 48 (L25, unverified) | Generous type. No assumption that older means no smartphone: "commonly said, unsupported" in [01-literature](../../fishermen-first/01-literature.md) Q9. |
| Data size | NDBC mobile page 2.6 KB; NWS text is small | Text-first status that loads without the map. A map as the opening screen is a bad trade on weak signal (inference) |
| Alerts | iOS web push only works after install to home screen; SMS in carrier review (`pilot/plan.md`) | Do not design the journey around a push that many cannot receive yet. |

---

## 9. Three evidence-based journeys

These are designed from the evidence above. They are not observed journeys.

### 1. First-time fisherman evaluating Neritic

- **Goal:** decide in one short visit whether this is worth coming back to. "Does it know my port and my fishery, and does it say something I'd otherwise spend time finding?"
- **Minimum information:** their port and fishery; the status for that fishery in plain words, from whom, dated; the next known decision date; the latest relevant test with its date; what Neritic does not know. For Dungeness today: "Commercial delayed to at least Jan 1, 2027, Zones 1–5, for humpback entanglements (CDFW). Next assessment expected Dec 15–18, subject to change."
- **Principal action:** **pick your port and fishery.** One choice, before anything else.
- **Outcome Neritic can reliably deliver:** a correct, dated, source-linked status and test summary, with the agency page one tap away. It must say plainly that notices are "not verified" by a person until reviewed. It cannot deliver a verdict on whether to go.
- **Honest risk:** the depth differs by fishery. Dungeness has the strongest content; squid, lobster, sablefish, urchin, spot prawn and albacore have no status record. A first visit in those fisheries should say so within seconds rather than show an empty dashboard.

### 2. Returning fisherman checking home port

- **Goal:** "Has anything changed, and what's the next few days?"
- **Minimum information:** what changed since last visit, by fishery and status; the next decision date; the zone forecast headline and days ahead with issue time; nearest live buoy with distance; the next tide; and, at bar ports, a link to the bar forecast and how to get the Coast Guard report.
- **Principal action:** **read what changed, and open its source.** If nothing changed, the page should say so at once.
- **Outcome Neritic can reliably deliver:** changes in status and tests; NWS text and buoy readings with their own times. It cannot deliver a go/no-go, a bar judgment, or anything better than NWS on weather.
- **Evidence behind it:** the wish to see many days ahead (Kuonen), trust in buoys (Kuonen), notice being the frustration (L18, Glickman). "Show changes first" is design inference: the repo's own J8 journey measured it as a regression in 2026-10 and no fisherman has tried either.

### 3. Researcher investigating a coastal observation or forecast

- **Goal:** establish what was observed or modelled at a place and time, and how far to trust it.
- **Minimum information:** variable and units; sensor or model; time and age of the value; distance from the place; whether it is observed, gap-filled or modelled; coverage and flags; skill against a baseline; a downloadable record.
- **Principal action:** **pick a place and time, read the record.** Then export.
- **Outcome Neritic can reliably deliver:** provenance-labelled observations and model values with their gaps shown, CSV exports (`v4/research/`), and the honest baseline: climatology + AR(1) held-out RMSE (ln) 0.760 against 0.857 for climatology and 0.892 for persistence (#74, n = 185k), chlorophyll only, 8-day period, not the bays. Also: the PINN matches, does not beat, the ConvLSTM, on the same pipeline. It cannot deliver a toxin forecast beyond C-HARM's three days, or independent validation.
- **Likely strongest audience.** Nothing in the competitor set serves this job in one place (inference from section 2 tools).

---

## 10. "Why would someone use Neritic tomorrow if they already have PredictWind, FishTrack and official agency websites?"

**The direct answer.** Not for wind and not for finding fish. PredictWind compares models and plans departures; FishTrack shows SST and chlorophyll for finding fish. Neritic should not try to beat either, and it has no evidence that it does.

The reason to open it tomorrow is narrow and real: **one phone screen tells you the current official state of your fishery at your port, the latest test numbers, the next date something can change, and what changed since you last looked, with the agency's own page one tap away.** None of the commercial apps does that (from their feature pages; I did not find any covering RAMP or domoic acid test results). Agency sites do hold all of it, but spread across at least eight hops for the crab opener (doc 02), in PDFs, without a statement of what is still undecided. Right now, with the commercial crab season held until at least Jan 1 and assessments due Dec 15–18, this is exactly the question the fleet is asking.

**Where the answer is weak**
1. **Agencies already push the news.** CDFW's GovDelivery topics cover domoic acid and whale-safe updates by email and text (doc 02). Neritic adds consolidation and plain order, not exclusivity.
2. **Peers and associations are faster and more trusted.** Associations negotiate price and sit on the working groups, so they likely hear first (inference). Colleagues are the most trusted source (S3). A new web tool is not.
3. **It is seasonal.** The value concentrates around crab decisions and closures. On an ordinary groundfish or lobster day, Neritic has little to say. The claim that weather, sea state and tide will bring people back daily (`pilot/plan.md` §1.4) is **untested** and rests on data they already get free.
4. **Fishery coverage is uneven.** Six curated status records exist; none for squid (largest by value), lobster, sablefish, urchin, spot prawn or albacore.
5. **Reach is uneven.** Four ports have mapped forecast zone, tide station and buoy; other ports are listed but I did not audit how much resolves for them. 2024's biggest port areas are Santa Barbara and Los Angeles.
6. **It has a credibility gap.** Notices are "Not verified" until a person reviews them. C-HARM has been reported stale. A fisherman burned once will not return (inference; Kuonen on trust lost through data disruptions).
7. **It may land in the middle of a distrust problem.** Fishermen's distrust of managers is strong (Glickman). Repackaging agency decisions could be read as neutral service or as an agency mouthpiece. Untested.
8. **Nobody has asked them.** Zero interviews. Everything here could be wrong.

**The most defensible position today:** a free, plain-language **tracker of what is open, tested and next**, starting with crab and toxin-affected fisheries, plus a serious research record. Weather supports the decision; it should not be the claim.

---

## 11. What interviews must settle

1. What is checked, in what order, on a normal morning? Does anything happen the night before?
2. How do you hear about an opener delay? Who told you first last time, and how long before the agency page?
3. Is a phone where this happens, or a laptop, a plotter, or the truck radio? (Do not ask if they'd use an app.)
4. Would a date and a "still undecided" list change a decision about crew, ice or travel? Which one?
5. What in a toxin result would you look at first? BMA? Site? Date? Average? Max?
6. What would make you stop trusting this?
7. For squid, lobster, sablefish, urchin: is there anything here at all, or does the question not arise?
8. Is there any secure way you'd share an observation with an agency or researcher, or none?
