"use client";

import type { PackRecord } from "./packs";
import { packAge, useOfflinePack } from "./useOfflinePack";
import { pacificToday } from "@/lib/time";
import { pacificDay } from "@/v4/ui/time";

/** What the service worker writes into a page it answers from a field pack (public/sw.js). */
type SavedMark = { at: string | null; url: string | null };

/** The mark of a page served from a field pack, or null (always null on the server). */
export function savedPage(): SavedMark | null {
  if (typeof window === "undefined") return null;
  const m = (window as unknown as { __NERITIC_SAVED__?: SavedMark }).__NERITIC_SAVED__;
  return m && typeof m === "object" ? m : null;
}

/**
 * The search params the server rendered a saved page with, when the page in the address bar is not
 * the one the pack saved (offline, `/coast?place=port-550` answers `/coast?place=port-550&for=commercial`
 * and every other place too). Hydration must render what that HTML holds, or React throws #418;
 * the page then renders the address bar's own place. Null when the two agree or online.
 */
export function savedSearch(pathname: string): URLSearchParams | null {
  const m = savedPage();
  if (!m?.url) return null;
  const u = new URL(m.url, window.location.origin);
  if (u.pathname !== pathname || u.search === window.location.search) return null;
  return u.searchParams;
}

export type SavedSnapshot = { savedAt: string; ageHours: number; outdated: boolean };

/** The pure rule behind useSavedSnapshot: a complete pack, and either offline or this page answered from it. */
export function savedSnapshot(record: PackRecord | null | undefined, online: boolean, now: Date = new Date()): SavedSnapshot | null {
  if (!record?.complete) return null;
  if (online && !savedPage()) return null;
  const { hours, outdated } = packAge(record, now);
  return { savedAt: record.savedAt, ageHours: hours, outdated };
}

/**
 * Whether what this view shows is a saved field-pack copy rather than live data, and when it was saved.
 * Null while online (unless the page itself came from the pack), when no complete pack covers the
 * place, and on the server and the first render (so it never changes hydrated text). Views use it to
 * date their headings ("As saved Oct 19, 3:07 PM") instead of saying "now".
 */
export function useSavedSnapshot(packId: string | null): { savedAt: string; ageHours: number; outdated: boolean } | null {
  const { record, online } = useOfflinePack(packId);
  return savedSnapshot(record, online);
}

/**
 * "30 h old", "3 days old", once a saved copy is outdated (more than a day); null before. Rounded as
 * the status centre's list rounds the same time (v4/ui/time relTime): whole hours up to a day and a
 * half, then Pacific calendar days from the time it was saved, so the banner and the list never
 * disagree by a day. Without `savedAt` the days are whole 24 h periods.
 */
export function ageWords(s: { ageHours: number; outdated: boolean; savedAt?: string } | null, now: Date = new Date()): string | null {
  if (!s?.outdated) return null;
  if (s.ageHours < 36) return `${Math.floor(s.ageHours)} h old`;
  const d = s.savedAt ? calendarDays(s.savedAt, now) : Math.floor(s.ageHours / 24);
  return d <= 1 ? "1 day old" : `${d} days old`;
}

/** Whole Pacific calendar days from an instant to now (as relTime counts them). */
function calendarDays(iso: string, now: Date): number {
  return Math.round((Date.parse(`${pacificToday(now)}T00:00:00Z`) - Date.parse(`${pacificDay(iso)}T00:00:00Z`)) / 86400000);
}
