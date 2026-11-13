import type { AlertTrigger } from "@/pilot/alerts/diff";
import { day } from "@/nx/ds/format";
import type { CheckReport } from "./check";
import type { CheckStamp, InboxItem } from "./inbox";

/**
 * The words alerts are said with (PLAN decision 3). One place, so the Alerts page, the Follow line on Today and
 * the tests say the same thing: what is checked, when, and what is never sent.
 */
export const DELIVERY = "Checked on this device when you open Neritic. Nothing is checked while it's closed. No texts, emails or phone notifications.";
export const DELIVERY_SHORT = "Checked when you open Neritic, on this device only.";
export const NOT_AVAILABLE = "Not available in this preview";
export const LOCAL_ONLY = "Everything here stays in this browser.";

/** What can alert the reader, in their words (a model is always named as a model). */
export const TRIGGER_WORDS: Record<AlertTrigger, { label: string; hint: string }> = {
  status_change: { label: "Season status and official notices", hint: "A closure, delay, restriction or advisory is added, changed or no longer listed" },
  new_toxin_result: { label: "New toxin test results", hint: "CDPH posts a new domoic acid result for a site near the port" },
  ramp_action: { label: "Whale-entanglement (RAMP) actions", hint: "CDFW changes a RAMP action or its next assessment date" },
  bloom_risk_jump: { label: "C-HARM probability jumps", hint: "NOAA's model probability rises sharply. A model forecast, not a toxin test" },
  data_stale: { label: "Records getting old", hint: "Neritic's copy of a source for this port is older than its update cycle" },
};

/** The ways an alert could reach the reader: one exists, the others are named so nobody assumes them. */
export const CHANNELS: Array<{ id: string; label: string; state: string; available: boolean }> = [
  { id: "in_app", label: "In Neritic, on this page", state: "On", available: true },
  { id: "push", label: "Phone or desktop notifications", state: NOT_AVAILABLE, available: false },
  { id: "sms", label: "Text messages", state: NOT_AVAILABLE, available: false },
  { id: "email", label: "Email", state: NOT_AVAILABLE, available: false },
];

/** An alert's title without the port in front (the page groups and names the port itself). */
export function itemTitle(i: Pick<InboxItem, "title" | "place_label">): string {
  const lead = `${i.place_label}: `;
  const t = i.title.startsWith(lead) ? i.title.slice(lead.length) : i.title;
  return t.replace(/^no longer listed — /, "");
}

/** What changed, in one line. */
export function itemChange(i: Pick<InboxItem, "change" | "was" | "trigger">): string {
  if (i.change === "gone") return "No longer listed here. The reason is unknown: check the official source before acting.";
  if (i.change === "changed" && i.was) return `Before: ${i.was}`;
  if (i.change === "new") return i.trigger === "new_toxin_result" ? "A new result." : "Newly listed for this port.";
  return "";
}

/** A sentence written by the pilot's rules, with its ISO days in the reader's terms ("since Oct 13"). */
export function plainDates(text: string, now?: string): string {
  return text.replace(/\b(\d{4}-\d{2}-\d{2})\b/g, (m) => {
    const t = Date.parse(`${m}T12:00:00Z`);
    return Number.isFinite(t) ? day(`${m}T12:00:00Z`, now ? { ref: now } : {}) : m;
  });
}

/**
 * What an alert's date is: a toxin result's date is its sample date; every other record's is when Neritic last read
 * it (the pipeline's check), not the agency's decision date, so it is never called "dated".
 */
export function asOfWord(i: { trigger: AlertTrigger }): "sampled" | "checked" {
  return i.trigger === "new_toxin_result" ? "sampled" : "checked";
}

/** The record's sentence for the page: ISO days in the reader's terms, and a quotation that ends a sentence ends it once (“….”. → “….”). */
export function recordWords(text: string, now?: string): string {
  return plainDates(text, now).replace(/([.!?])([”"’'])\./g, "$1$2");
}

/** What a check just did, in words for the page's status line. */
export function reportWords(r: CheckReport, name: (port: string) => string): string {
  switch (r.state) {
    case "nothing_followed":
      return "Follow a port to have it checked.";
    case "offline":
      return "You're offline: nothing was checked. The alerts already on this device are below.";
    case "throttled":
      return "Checked a moment ago: nothing new to read yet.";
    case "checked": {
      const parts: string[] = [];
      if (r.added.length) parts.push(`${r.added.length} new alert${r.added.length === 1 ? "" : "s"}.`);
      else if (r.stamp.ports.length) parts.push("Checked: nothing has changed.");
      if (r.baselined.length) parts.push(`${list(r.baselined.map(name))}: noted as ${r.baselined.length === 1 ? "it is" : "they are"} now. Changes from here on will be listed.`);
      if (r.stamp.failed.length) parts.push(`${list(r.stamp.failed.map(name))} could not be read just now and will be tried again.`);
      if (!r.persisted) parts.push("This browser is not keeping data, so this lasts only until you close the page.");
      return parts.join(" ");
    }
  }
}

/** "Last checked" for the page: when, and what could not be read. Null when this device has never checked. */
export function lastCheckedWords(stamp: CheckStamp | null, when: (iso: string) => string, name: (port: string) => string): string | null {
  if (!stamp || !stamp.at) return null;
  const failed = stamp.failed.length ? ` ${list(stamp.failed.map(name))} could not be read then.` : "";
  return `Last checked ${when(stamp.at)}.${failed}`;
}

export function list(xs: string[]): string {
  if (xs.length <= 1) return xs.join("");
  return `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`;
}
