/**
 * Pilot data loaders (Track D). Each sidecar is published outside the manifest and is optional:
 * a missing or invalid file is null and nothing else breaks.
 *  - biotoxin/index.json        -> loadBiotoxin        (CDPH domoic-acid tests, measured)
 *  - fisheries-status/index.json -> loadFisheriesStatus (curated official status + RAMP; review state)
 *  - derived/index.json         -> loadDerived          (8-day chlorophyll outlook, estimate, SST; lib/derived)
 */
export { loadBiotoxin, parseBiotoxin, sitesForPlace, overAction } from "./biotoxin";
export type { BiotoxinIndexExt, BiotoxinSiteExt, BiotoxinSetExt } from "./biotoxin";
export { loadFisheriesStatus, parseFisheriesStatus, recordsForPlace, rampZoneForPlace, reviewWords } from "./fisheriesStatus";
export type { FisheriesStatusIndexExt, FisheryStatusExt, RampStateExt, RampZoneExt } from "./fisheriesStatus";
export { loadDerived, parseDerived, derivedOutlook, derivedEstimate, derivedSst } from "@/lib/derived";
export type { DerivedIndex, OutlookLayer, EstimateLayer, SstLayer } from "@/lib/derived";
