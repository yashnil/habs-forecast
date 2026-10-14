import type { Snapshot } from "@/lib/briefingSnapshot";
import { AGENCY_LINKS } from "@/content/coast";
import { speciesById } from "@/content/species";
import { AUDIENCES } from "@/lib/coast";
import { formatDate } from "@/lib/time";

/**
 * A kept briefing: a dated copy of a snapshot that someone can print, or save as one self-contained
 * HTML file and open later without a connection. It keeps the time it was made, every original
 * agency, sample and model date, each source link written out in full, and the waters, activity and
 * species chosen. It always says it may be out of date and is never a legal authorization to fish or
 * harvest, or proof that seafood is safe.
 */
export const KEPT_KEY = "cw-kept-v1";
export const MAX_KEPT = 5;

export type Kept = { id: string; keptAt: string; snapshot: Snapshot };

export const KEPT_WARNING =
  "This is a saved copy, not current information. Agency actions after it was saved may replace anything in it. It is not a legal authorization to fish or harvest, and not proof that seafood is safe.";

export function makeKept(snapshot: Snapshot, now: Date): Kept {
  return { id: `${snapshot.place.key}-${now.getTime()}`, keptAt: now.toISOString(), snapshot };
}

export function readKept(): Kept[] {
  try {
    const v = JSON.parse(window.localStorage.getItem(KEPT_KEY) ?? "[]");
    return Array.isArray(v) ? v.filter((k) => k && typeof k.id === "string" && k.snapshot?.place) : [];
  } catch {
    return [];
  }
}

export function saveKept(k: Kept): boolean {
  try {
    window.localStorage.setItem(KEPT_KEY, JSON.stringify([k, ...readKept().filter((x) => x.id !== k.id)].slice(0, MAX_KEPT)));
    return true;
  } catch {
    return false;
  }
}

export function deleteKept(id: string): void {
  try {
    window.localStorage.setItem(KEPT_KEY, JSON.stringify(readKept().filter((x) => x.id !== id)));
  } catch {
    // nothing kept
  }
}

/** "Oct 13, 2026, 4:55 PM PDT": a kept copy can be read months later, so the year is always shown. */
export function stamp(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-US", { year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZone: "America/Los_Angeles", timeZoneName: "short" });
}

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
const day = (d: string | null) => (d ? formatDate(d, { year: true }) : "no date given");

export function activityLabel(a: string): string {
  return a === "all" ? "Any activity" : (AUDIENCES.find((x) => x.id === a)?.label ?? a);
}

export function keptTitle(k: Kept): string {
  const s = k.snapshot;
  return `${s.place.name} · ${activityLabel(s.audience)} · ${s.species.label}`;
}

export function keptFilename(k: Kept): string {
  const d = k.keptAt.slice(0, 10);
  return `coastwatch-${k.snapshot.place.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${d}.html`;
}

/** The plain sections of a kept briefing, shared by the in-app view and the saved file. */
export type KeptSection = { heading: string; lines: { text: string; url?: string; note?: string }[] };

export function keptSections(k: Kept): KeptSection[] {
  const s = k.snapshot;
  const o = s.official;
  const out: KeptSection[] = [];
  const species = speciesById(s.species.id);

  const status: KeptSection = { heading: "What the agencies say", lines: [] };
  if (!o.ok) status.lines.push({ text: "Official notices could not be loaded when this was saved. Nothing here is complete." });
  for (const st of o.statements) status.lines.push({ text: `${st.agency}: “${st.statement}”`, note: st.topic, url: st.url });
  const mine = o.notices.filter((n) => n.forYou && n.status === "active");
  for (const n of mine)
    status.lines.push({
      text: `${n.agency}: ${n.title}`,
      note: [n.published ? `${n.agency} dated ${day(n.published)}` : null, n.effective ? `effective ${day(n.effective)}` : null, n.tier === "check" ? "its geography does not settle whether it covers these waters" : null].filter(Boolean).join(" · ") || "no date given",
      url: n.url ?? undefined,
    });
  for (const d of o.disputes.filter((x) => o.notices.some((n) => n.id === x.recordId)))
    status.lines.push({ text: `${d.agency} ${d.release} appears to lift a notice above; not reviewed by a person`, url: d.url });
  if (o.ok && !mine.length && !o.statements.length)
    status.lines.push({ text: `No notice in CoastWatch's list covered these waters for ${species.id === "any" ? "any species" : s.species.label} when this was saved. That is not the same as open or safe.` });
  if (o.ok)
    status.lines.push({ text: `Transcribed by CoastWatch from agency pages; ${o.reviewStatus === "human_verified" ? "checked by a person" : "not checked by a person"}.`, note: `CoastWatch last checked the agencies' pages ${o.checkedAt ? stamp(o.checkedAt) : "(unknown)"}` });
  out.push(status);

  const others = o.notices.filter((n) => !n.forYou && n.status === "active");
  if (others.length) out.push({ heading: "Other notices in these waters (other species or activities)", lines: others.map((n) => ({ text: `${n.agency}: ${n.title}`, note: n.published ? `${n.agency} dated ${day(n.published)}` : "no date given", url: n.url ?? undefined })) });

  const meas: KeptSection = { heading: "Shore-station samples near these waters (seawater, not seafood)", lines: [] };
  for (const st of s.stations)
    meas.lines.push({ text: `${st.name}, ${Math.round(st.km)} km`, note: `toxin last sampled ${day(st.toxin)} · cell count last sampled ${day(st.cells)}`, url: st.url });
  if (!s.stations.length) meas.lines.push({ text: "No shore station within 60 km. No sample is not the same as no toxin." });
  out.push(meas);

  out.push({
    heading: "Model and data status",
    lines: [
      { text: s.forecast?.issued ? `C-HARM model run issued ${day(s.forecast.issued)} (a probability of bloom conditions, not a toxin measurement)` : "No C-HARM model run was available." },
      ...s.sources.filter((x) => x.outcome === "failed").map((x) => ({ text: `${x.title}: latest update failed`, note: x.lastSuccess ? `showing data from ${stamp(x.lastSuccess)}` : undefined })),
    ],
  });

  const check: KeptSection = { heading: "Check before you go", lines: [] };
  for (const h of o.hotlines) check.lines.push({ text: `${h.label}: ${h.phone}` });
  if (!o.hotlines.length) check.lines.push({ text: `${AGENCY_LINKS.cdfwClosureLine.label}: ${AGENCY_LINKS.cdfwClosureLine.phone}` }, { text: `${AGENCY_LINKS.cdphLine.label}: ${AGENCY_LINKS.cdphLine.phone}` });
  check.lines.push({ text: AGENCY_LINKS.cdfwAdvisories.label, url: AGENCY_LINKS.cdfwAdvisories.url }, { text: AGENCY_LINKS.cdphAdvisories.label, url: AGENCY_LINKS.cdphAdvisories.url });
  if (species.link) check.lines.push({ text: species.link.label, url: species.link.url });
  check.lines.push({ text: "NWS marine forecasts (weather is not part of this copy)", url: "https://www.weather.gov/marine/" });
  out.push(check);
  return out;
}

/** One self-contained HTML file (inline styles, no scripts, no external requests). */
export function keptHtml(k: Kept): string {
  const s = k.snapshot;
  const sections = keptSections(k)
    .map(
      (sec) =>
        `<section><h2>${esc(sec.heading)}</h2><ul>${sec.lines
          .map((l) => `<li><span class="t">${esc(l.text)}</span>${l.note ? `<span class="n">${esc(l.note)}</span>` : ""}${l.url ? `<a href="${esc(l.url)}">${esc(l.url)}</a>` : ""}</li>`)
          .join("")}</ul></section>`,
    )
    .join("");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${esc(`CoastWatch briefing, saved ${stamp(k.keptAt)}: ${keptTitle(k)}`)}</title>
<style>
:root{color-scheme:light}body{margin:0;background:#fff;color:#111;font:17px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:720px;margin:0 auto;padding:16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:20px 0 6px;border-bottom:2px solid #111;padding-bottom:2px}
.warn{border:3px solid #111;padding:10px 12px;margin:12px 0;font-weight:600}.meta{color:#333;font-size:15px}
ul{list-style:none;margin:0;padding:0}li{padding:8px 0;border-bottom:1px solid #ccc}.t{display:block}.n{display:block;color:#333;font-size:15px}
a{display:block;color:#0645ad;font-size:14px;word-break:break-all}
@media print{body{font-size:12pt}main{padding:0}a{color:#000}.warn{border-width:2px}@page{margin:14mm}}
</style></head><body><main>
<p class="meta">CoastWatch briefing, kept copy</p>
<h1>${esc(keptTitle(k))}</h1>
<p class="meta">Saved ${esc(stamp(k.keptAt))}${s.place.county ? ` · ${esc(s.place.county)} County` : ""} · ${esc(`${s.place.lat.toFixed(3)}°N, ${Math.abs(s.place.lon).toFixed(3)}°W`)}</p>
<p class="warn">${esc(KEPT_WARNING)}</p>
${sections}
<footer><p class="meta">Saved ${esc(stamp(k.keptAt))}. Built from CoastWatch data generated ${esc(stamp(s.dataGeneratedAt))}. CoastWatch is an independent research project, not a government agency. ${esc(KEPT_WARNING)}</p></footer>
</main></body></html>`;
}
