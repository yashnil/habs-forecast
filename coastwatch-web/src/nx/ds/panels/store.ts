import { validPlacement, type Box, type Placement } from "./geometry";

/**
 * Where the reader has put a page's panels, kept on this device only: one localStorage entry per page
 * (`nx-panels-v1:<page>`), holding each panel's placement, its size when resized and whether it is collapsed.
 * Every read and write is wrapped: private windows, blocked storage and a malformed entry all read as "nothing
 * moved", and the page lays itself out as designed.
 */
export const PANELS_VERSION = 1;
export const PANELS_NS = "nx-panels-v1:"; // PANELS_VERSION in the key (registered in src/nx/personal/keys.ts)

export type SavedPanel = { p?: Placement; w?: number; h?: number; c?: true };
export type SavedLayout = { v: typeof PANELS_VERSION; panels: Record<string, SavedPanel> };

type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">;
const storage = (): StorageLike | null => {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
};

const okSize = (n: unknown) => typeof n === "number" && Number.isFinite(n) && n > 0 && n < 10000;

/** One panel's entry, cleaned: unknown or malformed fields dropped. null when nothing usable is left. */
export function cleanPanel(x: unknown): SavedPanel | null {
  if (!x || typeof x !== "object") return null;
  const q = x as Record<string, unknown>;
  const out: SavedPanel = {};
  if (validPlacement(q.p)) out.p = { ax: q.p.ax, dx: Math.round(q.p.dx), ay: q.p.ay, dy: Math.round(q.p.dy) };
  if (okSize(q.w)) out.w = Math.round(q.w as number);
  if (okSize(q.h)) out.h = Math.round(q.h as number);
  if (q.c === true) out.c = true;
  return Object.keys(out).length ? out : null;
}

/** The page's saved layout; an entry of another version, or one that does not parse, reads as empty. */
export function parseLayout(raw: string | null): SavedLayout {
  const empty: SavedLayout = { v: PANELS_VERSION, panels: {} };
  if (!raw) return empty;
  try {
    const j = JSON.parse(raw) as { v?: unknown; panels?: unknown };
    if (j?.v !== PANELS_VERSION || !j.panels || typeof j.panels !== "object") return empty;
    const panels: Record<string, SavedPanel> = {};
    for (const [k, v] of Object.entries(j.panels as Record<string, unknown>)) {
      const c = cleanPanel(v);
      if (c && k.length <= 64) panels[k] = c;
    }
    return { v: PANELS_VERSION, panels };
  } catch {
    return empty;
  }
}

export function loadLayout(page: string, st: StorageLike | null = storage()): SavedLayout {
  try {
    return parseLayout(st?.getItem(PANELS_NS + page) ?? null);
  } catch {
    return { v: PANELS_VERSION, panels: {} };
  }
}

function write(page: string, layout: SavedLayout, st: StorageLike | null) {
  try {
    if (!st) return;
    if (!Object.keys(layout.panels).length) st.removeItem(PANELS_NS + page);
    else st.setItem(PANELS_NS + page, JSON.stringify(layout));
  } catch {
    // storage full or blocked: the layout simply is not kept
  }
}

/** Merges a panel's change into the page's entry (null fields removed); an emptied panel leaves the entry. */
export function savePanel(page: string, key: string, patch: { p?: Placement | null; w?: number | null; h?: number | null; c?: boolean }, st: StorageLike | null = storage()): SavedLayout {
  const layout = loadLayout(page, st);
  const cur: SavedPanel = { ...layout.panels[key] };
  if (patch.p !== undefined) {
    if (patch.p) cur.p = patch.p;
    else delete cur.p;
  }
  for (const k of ["w", "h"] as const) {
    const v = patch[k];
    if (v === undefined) continue;
    if (v != null && okSize(v)) cur[k] = Math.round(v);
    else delete cur[k];
  }
  if (patch.c !== undefined) {
    if (patch.c) cur.c = true;
    else delete cur.c;
  }
  const clean = cleanPanel(cur);
  if (clean) layout.panels[key] = clean;
  else delete layout.panels[key];
  write(page, layout, st);
  bump();
  return layout;
}

/** Forgets every panel's placement, size and collapse on the page. */
export function resetLayout(page: string, st: StorageLike | null = storage()) {
  try {
    st?.removeItem(PANELS_NS + page);
  } catch {
    // nothing kept to forget
  }
  resets.set(page, (resets.get(page) ?? 0) + 1);
  bump();
}

export const layoutChanged = (l: SavedLayout) => Object.keys(l.panels).length > 0;

/* ---------------------------------------------------------------- the live state the panels share */

/** The page whose panels are free (PageLayout `panels`); null: the stage objects stand where the page puts them. */
let page: string | null = null;
let rev = 0;
const resets = new Map<string, number>();
const listeners = new Set<() => void>();
function bump() {
  rev++;
  listeners.forEach((l) => l());
}
export function subscribePanels(l: () => void) {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}
export const panelsRev = () => rev;
export const getPanelsPage = () => page;
export const resetCount = (p: string) => resets.get(p) ?? 0;

/** Called by PageLayout: the page's panels are free (or not) while it is mounted. */
export function setPanelsPage(p: string | null) {
  if (page === p) return;
  page = p;
  bump();
}

/** The panels now on the stage: each one's box where it stands and where it would rest unmoved (the default layout). */
type Live = { el: HTMLElement; moved: boolean; rest: Box | null };
const live = new Map<string, Live>();
export function registerPanel(key: string, el: HTMLElement) {
  live.set(key, { el, moved: false, rest: null });
  return () => {
    if (live.get(key)?.el === el) live.delete(key);
  };
}
export function notePanel(key: string, patch: Partial<Omit<Live, "el">>) {
  const cur = live.get(key);
  if (cur) Object.assign(cur, patch);
}
export const livePanels = () => [...live.entries()].map(([key, v]) => ({ key, ...v }));
