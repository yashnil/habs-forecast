# 03 · CO-OPS harbour water temperature: audit (2026-10-15)

What the `harbor` part of `/api/conditions` may say, checked against NOAA's own documentation and metadata.
Code: `coastwatch-web/src/lib/conditions.ts` (parser, limits, candidate rule), `src/app/api/conditions/route.ts`
(`harborPart`), `src/content/stations/coops-watertemp-ca.json` (+ `README.md`), `tests/unit/conditions.test.ts`.

## 1. Quality flags (`f`)

Source: https://api.tidesandcurrents.noaa.gov/api/prod/ ("Data API Response Descriptions" → Response Help Page)
→ https://api.tidesandcurrents.noaa.gov/api/prod/responseHelp.html, section **Water Temperature**, read 2026-10-15:
"Data Flags - in order of listing: (X) … the maximum expected water temperature was exceeded; (N) … the minimum
expected water temperature was exceeded; (R) … the rate of change tolerance limit was exceeded." Each is "1"
when set. So `"0,0,0"` = all three checks passed; `"1,0,0"` = too warm for the station's expected range; `"0,1,0"` =
too cold; `"0,0,1"` = jumped faster than allowed. (Air temperature, pressure, humidity and conductivity use the
same X,N,R triple; water level uses a different string, O/F/R/L, not relevant here.) NOAA's thermistor accuracy is
not stated on these pages; the ±0.2 °C quoted in 01-sources row E is not repeated in code.

Implementation: `WATER_TEMP_FLAGS` (X, N, R in order), `waterTempFlagsPass` (any non-"0" in any position rejects,
including an unknown fourth position), `waterTempFlagNote` (words for the UI). `HarborTemp` gains optional
`flagNote` ("passed NOAA's checks: maximum, minimum, rate of change" or "set aside: …"). Tested per position.

## 2. Sensor placement and depth

Sources, read 2026-10-15: https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=watertemp
(16 California stations, identical to the bundled list) and, per station,
`https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{id}/sensors.json` and `/details.json`.
`details.json` holds only establishment dates, chart number and time zone: **no depth, no placement**.
`sensors.json` gives each sensor an `elevation` in **feet relative to a datum** (MLLW for the thermistor E1) and a
`status`. Nothing states the horizontal position on the pier or the depth below the surface; depth below the
surface is the tide height minus the elevation below MLLW. What it states:

| Station | Thermistor E1 elevation | Status (2026-10-15) |
|---|---|---|
| 9410170 San Diego | −18.2 ft (5.6 m below MLLW) | active |
| 9410230 La Jolla | −11.3 ft (3.4 m) | active |
| 9410660 Los Angeles | **no E1 sensor listed** (only pressure, tsunami and microwave water level) | answers "No data" |
| 9410840 Santa Monica | −9.3 ft (2.8 m) | active |
| 9412110 Port San Luis | −8.5 ft (2.6 m) | active |
| 9413450 Monterey | −6.8 ft (2.1 m) | active |
| 9414290 San Francisco | −3.8 ft (1.2 m) | **inactive since 2025-12-10**: "Suspect Data - Data failed to meet QC standards - under review" |
| 9414523 Redwood City | −3.0 ft (0.9 m) | inactive since 2026-03-05 (same message) |
| 9414750 Alameda | −3.9 ft (1.2 m) | active |
| 9414863 Richmond | −4.1 ft (1.3 m) | active |
| 9415020 Point Reyes | −8.4 ft (2.6 m) | active |
| 9415102 Martinez-Amorco Pier | −6.0 ft (1.8 m) | inactive since 2025-12-16 (same message) |
| 9415144 Port Chicago | −1.0 ft (0.3 m) | active |
| 9416841 Arena Cove | −5.5 ft (1.7 m) | active |
| 9418767 North Spit | −3.1 ft (1.0 m) | active |
| 9419750 Crescent City | −3.0 ft (0.9 m) | active |

Santa Barbara 9411340 (34.40459, −119.6925; 0.5 km from the harbour point,
https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/9411340.json) was checked too: not under
`type=watertemp`, no E1 in its sensor list, and the datagetter for the test window answers "No data"; it stays out. Each bundled entry
now carries `sensorNote` with the elevation, the status and the date read; the route copies it into `station.sensorNote`.

## 3. Representativeness: the rule

A thermistor is used **only if it is the nearest water-temperature station and within 6 km** (`HARBOR_TEMP_MAX_KM`,
harbour/bay scale; was 20). Within 2 km (`HARBOR_TEMP_AT_KM`) the part says `basis: "in-harbour"`, otherwise
`basis: "nearby-station"` and the UI names the station with its distance. **No fallback to a second station**:
if the nearest one is down the part is `unavailable` ("… sensor not reporting"), never a reading from across the
bay. Beyond 6 km the part is `none` ("No NOAA water-temperature station within 6 km."). Before this change San
Francisco (37.809, −122.416) fell back to Alameda 9414750, 10.9 km away inside the bay.

Outcome for the 22 ports (`public/data/v1/ports-adfcb13636.geojson`), recorded window 2026-10-13 03:00–06:00 GMT:

| Port | Nearest thermistor | Result |
|---|---|---|
| Crescent City | 9419750, 1.7 km | ok, in-harbour (11.0 °C) |
| Point Arena | 9416841 Arena Cove, 1.7 km | ok, in-harbour (12.8 °C) |
| Monterey | 9413450, 1.0 km | ok, in-harbour (17.6 °C) |
| Avila/Port San Luis | 9412110, 0.7 km | ok, in-harbour (18.2 °C) |
| Eureka | 9418767 North Spit, 5.7 km | ok, nearby-station (13.1 °C) |
| San Diego | 9410170, 4.0 km | ok, nearby-station (24.2 °C) |
| San Francisco | 9414290, 4.4 km | unavailable: sensor not reporting (Alameda 10.9 km, Richmond 13.4 km are not used) |
| San Pedro | 9410660 Los Angeles, 2.6 km | unavailable: sensor not reporting |
| Trinidad (33 km), Shelter Cove, Fort Bragg, Bodega Bay (38 km), Princeton-Half Moon (24 km), Santa Cruz (42 km), Moss Landing (24 km), Morro Bay (24 km), Santa Barbara, Ventura, Port Hueneme, Newport Beach (34 km), Dana Point, Oceanside (39 km) | none within 6 km | none |

## 4. Freshness and outages

The series is 6-minute and normally reaches the API 6–12 min after the observation (01-sources row E), but a
station's transmission or CO-OPS's processing can lag, so readings up to **3 h** old are accepted
(`HARBOR_TEMP_MAX_AGE_H`) rather than reporting a working station as silent. A reading older than **1 h** is
returned with `stale: true` (`HARBOR_TEMP_STALE_H`) so the UI can say "last reading 2 h ago". Readings more than
an hour in the future (clock skew allowance) are set aside. Verified by unit tests: no rows → unavailable "no
readings in the last 3 h"; all rows flagged → unavailable "every reading … failed NOAA's quality checks"; the
`{"error":"No data was found…"}` object → "sensor not reporting"; a listed station whose response is missing
from the recording (treated as an outage, like HTTP 500) → "the CO-OPS service did not answer (…)"; future
timestamps → set aside. The parser's `parseWaterTempSeries` counts rows, flagged and out-of-window rows so the
route can say which case it is.

## 5. Fixtures

The 16 recorded datagetter responses (one per station, sha1 of the URL + ".json", window
`begin_date=20261013 03:00&end_date=20261013 06:00`, GMT) are unchanged; the request URL shape is unchanged; no
station was added. Route tests run `GET` against them with the fixed clock 2026-10-13T06:00:00Z.
