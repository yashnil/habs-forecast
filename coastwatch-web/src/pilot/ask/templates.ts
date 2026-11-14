import type { PilotPort } from "./places";
import type { Referral, SafetyKind } from "./types";

/**
 * Fixed wording. Nothing here is generated: safety answers, the disclaimer and the referrals are
 * the same in demo and live mode, on the web and by SMS.
 */

export const DISCLAIMER =
  "Neritic is not an official source. Closures, seasons and seafood advisories come from CDFW and CDPH and always take precedence. Chlorophyll is not toxin, and forecasts are probabilities, not guarantees. Never use Neritic to decide whether seafood is safe or whether it is safe to go out.";

/**
 * The standing notice under the first answer of a conversation: short, and after the answer (the full
 * DISCLAIMER stands on the page itself). What it must always say: not an official source, the agencies
 * take precedence, never for deciding whether seafood is safe or whether to go out.
 */
export const ANSWER_NOTICE = "Neritic is not an official source: CDFW and CDPH decide, and their notices take precedence. Never use Neritic to decide whether seafood is safe or whether it is safe to go out.";

export const DEMO_LABEL = "Pilot demo — answers are assembled from Neritic's data without an AI model";
export const LIVE_LABEL = "Live — answers are written by Claude from Neritic's data and checked before you see them";
export const SAMPLE_DATA_LABEL = "Sample data — the pilot's fixture records, not today's published data";

export const REF = {
  cdphShellfish: { label: "CDPH shellfish & biotoxin advisories", url: "https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" },
  cdphDomoic: { label: "CDPH domoic acid results", url: "https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Domoic-Acid.aspx" },
  cdfwAdvisories: { label: "CDFW health advisories & closures", url: "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" },
  cdfwWhale: { label: "CDFW whale-safe fisheries (RAMP)", url: "https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries" },
  cdfwInseason: { label: "CDFW in-season regulation changes", url: "https://wildlife.ca.gov/Fishing/Ocean/Regulations/inseason" },
  nwsMarine: { label: "NWS marine forecasts", url: "https://www.weather.gov/marine/" },
  uscg: { label: "U.S. Coast Guard Navigation Center", url: "https://www.navcen.uscg.gov/" },
} as const;

export function nwsZone(port: PilotPort | null): Referral {
  return port
    ? { label: `NWS marine forecast, zone ${port.nws_zone}`, url: `https://forecast.weather.gov/MapClick.php?zoneid=${port.nws_zone}`, why: "Official marine forecast for this water" }
    : { ...REF.nwsMarine, why: "Official marine forecasts" };
}

export type SafetyTemplate = { lines: string[]; referrals: Referral[]; sms: string };

export function safetyTemplate(kind: SafetyKind, port: PilotPort | null): SafetyTemplate {
  switch (kind) {
    case "eat":
      return {
        lines: [
          "Neritic doesn't decide whether seafood can be eaten, and it never says that it can.",
          "CDPH decides that from laboratory tests, and CDFW sets closures. Check them before you eat or sell anything you catch or gather.",
          "What Neritic has on record is below, for information only.",
        ],
        referrals: [
          { ...REF.cdphShellfish, why: "Whether seafood from this area is under an advisory" },
          { ...REF.cdphDomoic, why: "The latest lab results" },
          { ...REF.cdfwAdvisories, why: "Fishery closures and health advisories" },
        ],
        sms: "Neritic can't say whether seafood can be eaten. CDPH decides: cdph.ca.gov (Shellfish Advisories); closures: wildlife.ca.gov.",
      };
    case "go_out":
      return {
        lines: [
          "Neritic can't tell you whether it is safe to go out; that is your call as skipper.",
          "Check the official NWS marine forecast and any small craft advisory before you leave. In an emergency, call the Coast Guard on VHF channel 16.",
          "Neritic's sea and weather records are below, for information only, not a go/no-go.",
        ],
        referrals: [nwsZone(port), { ...REF.uscg, why: "Coast Guard notices to mariners" }],
        sms: "Neritic can't say if it's safe to go out. Check the NWS marine forecast (weather.gov/marine); emergency: USCG VHF 16.",
      };
    case "bar":
      return {
        lines: [
          "Neritic has no bar or harbor-entrance data and can't tell you whether a crossing is safe.",
          "Bar and entrance conditions come from the U.S. Coast Guard and the harbor office; call them, and in an emergency use VHF channel 16.",
          "Neritic's sea and weather records are below, for information only.",
        ],
        referrals: [{ ...REF.uscg, why: "Bar restrictions and notices to mariners" }, nwsZone(port)],
        sms: "Neritic has no bar data. Ask the US Coast Guard or harbor office; emergency: VHF 16. Forecast: weather.gov/marine.",
      };
    case "fish_legal":
      return {
        lines: [
          "Neritic isn't an official source and can't tell you that you may fish, set gear or sell.",
          "CDFW decides seasons, closures and gear rules, and CDPH decides seafood advisories. Check CDFW before you go.",
          "The records Neritic has are below, for information only.",
        ],
        referrals: [
          { ...REF.cdfwInseason, why: "Season and in-season changes" },
          { ...REF.cdfwWhale, why: "Whale-entanglement (RAMP) actions" },
          { ...REF.cdfwAdvisories, why: "Health advisories and closures" },
        ],
        sms: "Neritic can't say if you may fish. CDFW decides seasons & closures: wildlife.ca.gov.",
      };
  }
}

export const HELP_LINES = [
  "Ask about one pilot port: what is open or delayed, the latest toxin tests, sea, wind and tide, the bloom outlook, whale rules, what changed, or where the data comes from.",
  "Every answer lists its sources and how old the data is.",
];

export const STANDING_REFERRALS: Referral[] = [
  { ...REF.cdfwAdvisories, why: "Closures and health advisories" },
  { ...REF.cdphShellfish, why: "Seafood advisories" },
];
