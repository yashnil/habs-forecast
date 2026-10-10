# Draft email to NOAA CoastWatch (not sent)

**To:** NOAA CoastWatch West Coast Node / central CoastWatch ERDDAP help desk. Use the contact address listed on <https://coastwatch.pfeg.noaa.gov/erddap/information.html> and <https://coastwatch.noaa.gov/cwn/contact.html>; check both pages for the current address before sending.
**Subject:** Automated access from GitHub Actions: intermittent HTTP 403, "Currently unknown datasetID" and 502 on ERDDAP (C-HARM, OLCI sectors, VIIRS, HF radar)

---

Hello CoastWatch team,

I maintain CoastWatch California (a non-commercial, open-source project, <https://github.com/yashnil/habs-forecast>). It republishes several of your ERDDAP products, with attribution and unchanged values, for a public coastal-conditions map:

- C-HARM v3.1 (`wvcharmV3_0day` … `wvcharmV3_3day`);
- the Sentinel-3 OLCI chlorophyll sectors (`noaacwS3[AB]OLCIchlaSector{CI,DI}Daily`);
- VIIRS chlorophyll (`erdVHNchla1day`);
- HFRNet surface currents (`ucsdHfrW2`).

A scheduled job runs every 6 hours on GitHub-hosted Actions runners (Ubuntu, Azure IP ranges). Each run makes about 30 data requests and up to about 250 small point queries, which verify our published values against yours. It sends `User-Agent: CoastWatch-pipeline/0.1 (+https://github.com/yashnil/habs-forecast)`. It retries 403/429/5xx up to 3 times with backoff, and it stops calling a dataset for the rest of the run after two exhausted failures.

On 2026-10-09/10 we saw three kinds of failure. All times are UTC.

**1. HTTP 403 (empty body) from `coastwatch.pfeg.noaa.gov`, GitHub runners only.**
The same URLs returned 200 from a residential connection minutes later.

| Run (start → end) | Datasets refused for the whole run (4 attempts each) |
|---|---|
| 2026-10-09 17:08 → 17:12 | `wvcharmV3_0day` … `_3day` |
| 2026-10-09 17:37 → 18:25 (errors logged 18:23:53) | `wvcharmV3_0day` … `_3day`, `erdVHNchla1day` (while `noaacwS3AOLCI*` on `coastwatch.noaa.gov` worked) |
| 2026-10-09 22:40 → 22:51 (errors logged 22:51:00) | `wvcharmV3_0day` … `_3day`, `erdVHNchla1day` |

Example request: `https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_0day.csv0?time[last]`

**2. HTTP 404 "Currently unknown datasetID".**

| Time | Datasets |
|---|---|
| 2026-10-09, during the day, for over an hour | `noaacwS3AOLCIchlaSectorCIDaily` and the other OLCI sectors on `coastwatch.noaa.gov` (also HTTP 502 in that period) |
| 2026-10-10 00:18–00:51 (observed from both GitHub and a desktop) | `noaacwS3[AB]OLCIchlaSector{CI,DI}Daily` on `coastwatch.noaa.gov`; `ucsdHfrW2` on `coastwatch.pfeg.noaa.gov` (from about 00:35; it disappeared between our download and our verification queries) |

Both had returned by 00:51 UTC.

**3. HTTP 502 "Proxy Error" from `coastwatch.noaa.gov`.**
At 2026-10-09 22:40–22:51, for the time listings of all four OLCI sector datasets, for example:
`https://coastwatch.noaa.gov/erddap/griddap/noaacwS3AOLCIchlaSectorCIDaily.csv0?time[(2026-10-02T00:00:00Z):1:(last)]`

**We also saw:**
- one run (2026-10-09 23:26 → 2026-10-10 00:17) stalled for about 50 minutes without completing. We now cap each download at 240 s.
- ERDDAP snapping a time-range start bound to the nearest time, so `time[(2026-10-02T00:00:00Z):…]` returned 2026-10-01T19:19Z. We now filter this on our side. We mention it only in case it isn't the intended behaviour.

**Questions:**
1. Are requests from cloud CI address ranges, such as GitHub Actions / Azure, rate-limited or filtered? If so, is there a recommended practice: a registered User-Agent or contact header, an allow-list, a request-rate ceiling, or preferred hours?
2. Is there a recommended operational endpoint for automated, recurring access to these datasets, for example a mirror, the AWS/NODD copies, or THREDDS/OPeNDAP? Our attempts to reach `hfrnet-tds.ucsd.edu` timed out.
3. Is "Currently unknown datasetID" expected during dataset reloads? If so, how long do reloads usually take, and is there a status page or feed we should check before treating it as an outage?
4. Would you prefer we reduce our verification point queries? We can switch to a few bounding-box requests per run.

We keep the last valid data, with their original dates, whenever your servers don't answer, and we never present our copies as newer than they are. Thank you for these datasets; they make a real difference to how coastal conditions can be communicated.

Best regards,
[Name]
CoastWatch California, <https://github.com/yashnil/habs-forecast>

---

**Before sending:**
- confirm the contact address on the two NOAA pages above;
- add your name and preferred reply address (not filled in here);
- attach or link the run logs if NOAA asks: <https://github.com/yashnil/habs-forecast/actions/runs/37967595146>, <https://github.com/yashnil/habs-forecast/actions/runs/38000436793>, <https://github.com/yashnil/habs-forecast/actions/runs/38009420295>.
