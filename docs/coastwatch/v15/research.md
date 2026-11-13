# v15 trip companion — desk research note

Desk research only (2026-11-13). No interviews. Anything marked *Inference* is our reading, not a sourced finding.

## Problems this feature solves

- **Go / no-go is a weather-and-bar call first, economics second.** In a November 2010 dockside survey of 83 Newport, Oregon Dungeness crab fishermen (24 of them captains), the factors rated "very important" in deciding to cross a river bar were timing of tides (96%), weather reports (83%), height of tide (67%), time of season (39%), amount of crab caught (21%) and time since pots were last tended (17%). The same report says most of the fishery's fatalities (79%, 14 deaths in 2003–2009) came from vessels capsizing while crossing river bars or working near shore [1] (abstract, checked against the PDF). NIOSH's West Coast summary says crews should avoid hazardous bar crossings and call the Coast Guard when a crossing can't be avoided [2].
- **Risk tolerance depends on the vessel, crew and money.** Across six US federal fisheries, aversion to storms falls as vessel size rises and grows with the severity of the NWS warning [3]. Cornish skippers go out more as wind and waves rise, up to a threshold. Skippers with crew "push the weather" more, and they combine digital forecasts, real-time observations and what they see on the water [4].
- **The data is scattered.** Forecasts, buoys, tides, closures and prices sit in separate places. *Inference:* one view that puts the departure window, the bar tide and the regulatory status side by side saves real time every morning.

## What competitors already supply

- **PredictWind Departure Planning** "compares four departure dates and associated weather conditions" along a route, across forecast models the user selects; roll/slamming metrics are Professional-only. Its feature page does not state the tier for the base feature. It is built for passage routing, not fishing economics [5][6].
- **FishTrack** offers cloud-free SST, chlorophyll from VIIRS and MODIS, sea-surface height, currents and bathymetry overlays, updated several times a day. It is aimed at offshore recreational and charter anglers [7][8]. We found no trip economics or logbook in its public material.
- **Catchwise** (catchwise.com, Oslo): "Catch history, satellite AIS and ocean data. Built for commercial fishing": catch logs and haul history, AIS vessel tracking, marine weather and ocean layers, and catch forecasts; 14-day trial, "used by 400+ skippers" [14]. It predicts catch; v15 deliberately does not.
- **Regulatory logbooks.** CDFW requires electronic fish tickets through E-Tix (run by PSMFC) for all commercial landings since 1 July 2019. The **fish business** usually submits the ticket within 3 business days [9][10]. Fishery logbooks are still largely paper from CDFW, and CPFVs are encouraged to use E-Logs [9]. Deckhand Pro offers offline e-logbooks and reporting to regulators [11]. Fishcom advertises logbook, finances and forecasts for commercial fishers [12].
- **Gap.** *Inference:* none of the products we found combines official NWS, bar, tide and closure data, domoic-acid tests and a per-trip net-to-boat figure.

## Implications for v15

1. **Lead with the departure window, not the calculator.** Put the NWS zone periods, the nearest NDBC buoy and the bar tide on one time axis, against the skipper's own limits; Neritic marks where a stated value crosses a limit and never labels a period favourable. The sources rank these above economics [1][3].
2. **Bar crossing as its own line.** Show the next flood and ebb at the home bar's CO-OPS station, and the bar-related wording from the NWS forecast where present. Do not issue a "safe" verdict. Link to the USCG and NIOSH guidance instead [2].
3. **Personal thresholds.** Let the skipper set their own wind and sea limits per vessel, because tolerance varies with vessel size and crew [3][4]. Present these as the skipper's own limits, never as advice.
4. **Show the forecast as issued.** Keep forecasts, observations and closures clearly separate, in line with the project rule that observations are never presented as forecasts. Show the issue time.
5. **Make economics a saved trip record.** Save each trip as a record: date, port, species, hours, fuel used, price paid. Pre-fill the next trip from the last one and show running season totals. *Inference:* tools stay in use when the second trip costs less effort than the first and they hold a history the user can't easily rebuild. A blank calculator gets used once.
6. **Settle the trip after landing.** Offer a "settle trip" step for entering the actual pounds and price from the landing receipt. Fishermen and businesses have E-Tix landing receipts [9][10], but we found no evidence of a public consumer import format. *Inference:* start with manual entry and an optional photo of the receipt, and plan a CSV import later.
7. **Fuel burn.** *Inference:* most skippers know an approximate gallons-per-hour or gallons-per-trip figure and few have engine telemetry. Ask for gal/h × hours with a per-vessel default, and don't require NMEA integration.
8. **Prices.** Treat buyer prices as user-entered and private. Do not crowdsource or display other people's prices without an explicit decision. *Inference:* dock prices are negotiated and sensitive.
9. **Privacy by default.** Store trip records on the device or in a private account. No social sharing. Never collect GPS fishing spots, and keep location to port level. California treats landing-receipt data as confidential (Fish & Game Code §8022; *we did not re-read this for this note, so verify it*). App precedents include opt-in sharing (eCatch) and device-only logs [13]. State this plainly in the UI.
10. **Regulatory status on every trip.** Attach the CDFW season, RAMP and CDPH domoic-acid status in force on the trip date to the saved record, labelled "unverified" where it is. Never present chlorophyll as toxin.

## Sources

1. Oregon crab fishing safety assessment (UW DEOHS): https://deohs.washington.edu/sites/default/files/images/general/FINAL_Oregon_Crab_Fishing_Safety_Assess_with_cover.pdf
2. NIOSH, Commercial Fishing Fatality Summary, West Coast 2015–2019: https://www.cdc.gov/niosh/docs/2025-113/pdfs/2025-113.pdf ; NIOSH capsizing case: https://stacks.cdc.gov/view/cdc/164886
3. "How storms affect fishers' decisions about going to sea," ICES JMS 2020: https://repository.library.noaa.gov/view/noaa/62949
4. Exeter, Cornish skippers (Global Environmental Change 2021): https://news-archive.exeter.ac.uk/2021/march/articles/fishersatriskinperfectsto.html ; https://ore.exeter.ac.uk/repository/handle/10871/125193
5. PredictWind Departure Planning: https://predictwind.com/features/departure-planning
6. PredictWind help: https://help.predictwind.com/en/articles/2884536-how-to-use-the-departure-planning
7. FishTrack charts: https://fishtrack.com/fishing-charts
8. FishTrack chlorophyll: https://support.fishtrack.com/hc/en-us/articles/4573253820443-Chlorophyll
9. CDFW MFSU: https://wildlife.ca.gov/Conservation/Marine/MFSU ; Landing resources: https://wildlife.ca.gov/Fishing/Commercial/Landing-Resources
10. PSMFC E-Tix: https://www.psmfc.org/program/electronic-fish-tickets-etix/
11. Deckhand Pro: https://apps.apple.com/us/app/deckhand-pro/id1494982202
12. Fishcom: https://apps.apple.com/us/app/fishcom/id6761494150
13. eCatch Mobile: https://apps.apple.com/app/id1061243164
14. Catchwise: https://www.catchwise.com
