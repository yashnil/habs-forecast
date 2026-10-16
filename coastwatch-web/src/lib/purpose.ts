"use client";

import { useCallback, useEffect, useState } from "react";
import type { Audience } from "@/lib/coast";

/**
 * Why someone uses Neritic, chosen once (optional) and kept on this device only. It changes
 * only defaults and order: which kind of fishing My Coast starts filtered to, and which section
 * comes first for researchers. It never changes a value, a notice, a caveat or a verification
 * status, and every page stays available to everyone. Nothing identifying is stored.
 */
export type Purpose = "commercial" | "recreational" | "research" | "ports" | "exploring";
export type PurposeChoice = { purpose: Purpose; harvest?: "fishing" | "shellfish" };

export const PURPOSES: { id: Purpose; label: string; hint: string }[] = [
  { id: "commercial", label: "Commercial fishing", hint: "Notices and conditions for the waters you fish" },
  { id: "recreational", label: "Recreational fishing or shellfish harvest", hint: "Rules and advisories where you fish or gather" },
  { id: "research", label: "Research and science", hint: "Measurements, model coverage and the gaps" },
  { id: "ports", label: "Ports and coastal operations", hint: "One port's notices and data ages at a glance" },
  { id: "exploring", label: "Just exploring", hint: "A short guided tour of the map" },
];

export const PURPOSE_KEY = "cw-purpose-v1";
const EVENT = "cw-purpose-change";

export function readPurpose(): PurposeChoice | null {
  try {
    const v = window.localStorage.getItem(PURPOSE_KEY);
    if (!v) return null;
    const p = JSON.parse(v) as PurposeChoice;
    return PURPOSES.some((x) => x.id === p.purpose) ? { purpose: p.purpose, harvest: p.harvest === "shellfish" ? "shellfish" : p.harvest === "fishing" ? "fishing" : undefined } : null;
  } catch {
    return null;
  }
}

export function writePurpose(p: PurposeChoice | null) {
  try {
    if (p) window.localStorage.setItem(PURPOSE_KEY, JSON.stringify(p));
    else window.localStorage.removeItem(PURPOSE_KEY);
  } catch {
    // not remembered on this device
  }
  window.dispatchEvent(new Event(EVENT));
}

/** The saved purpose, kept in sync across components on the page. null until read (and when none). */
export function usePurpose(): [PurposeChoice | null, (p: PurposeChoice | null) => void, boolean] {
  const [p, setP] = useState<PurposeChoice | null>(null);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const on = () => setP(readPurpose());
    on();
    setReady(true);
    window.addEventListener(EVENT, on);
    window.addEventListener("storage", on);
    return () => {
      window.removeEventListener(EVENT, on);
      window.removeEventListener("storage", on);
    };
  }, []);
  const set = useCallback((x: PurposeChoice | null) => writePurpose(x), []);
  return [p, set, ready];
}

/** My Coast's starting filter for a purpose (a link's own filter always wins). */
export function defaultAudience(p: PurposeChoice | null): Audience {
  if (!p) return "all";
  if (p.purpose === "commercial") return "commercial";
  if (p.purpose === "recreational") return p.harvest === "shellfish" ? "shellfish" : "recreational";
  return "all";
}

/** Researchers start with what was measured and where the model has values; official notices stay in the summary. */
export type CoastSection = "official" | "measured" | "model" | "observed";
export function sectionOrder(p: PurposeChoice | null): CoastSection[] {
  return p?.purpose === "research" ? ["measured", "model", "observed", "official"] : ["official", "measured", "model", "observed"];
}

export function purposeLabel(p: PurposeChoice | null): string {
  if (!p) return "No preference";
  const base = PURPOSES.find((x) => x.id === p.purpose)!.label;
  if (p.purpose === "recreational" && p.harvest) return p.harvest === "shellfish" ? "Shellfish harvest" : "Recreational fishing";
  return base;
}
