/**
 * Which link clicks get a route transition. Pure: tested on its own (tests/unit/motion-primitives.test.ts).
 * In-app means the same origin and another page: a query-only change on the same page is that page's own state
 * (a port switch, a layer) and cuts; a hash-only change is a scroll.
 */
export type ClickLike = { button: number; metaKey: boolean; ctrlKey: boolean; shiftKey: boolean; altKey: boolean; defaultPrevented?: boolean };
export type AnchorLike = { href: string | null; target?: string | null; download?: boolean; noTransition?: boolean };
export type LocationLike = { origin: string; pathname: string };

const NOT_PAGES = /^\/(api|_next|data|basemap|vendor|fonts)\//;
const FILE_LIKE = /\.[a-z0-9]{2,6}$/i;

export function isRouteChange(href: string, loc: LocationLike): boolean {
  let u: URL;
  try {
    u = new URL(href, loc.origin + loc.pathname);
  } catch {
    return false;
  }
  if (u.protocol !== "http:" && u.protocol !== "https:") return false;
  if (u.origin !== loc.origin) return false;
  if (u.pathname === loc.pathname) return false;
  return !NOT_PAGES.test(u.pathname) && !FILE_LIKE.test(u.pathname);
}

export function shouldIntercept(click: ClickLike, a: AnchorLike, loc: LocationLike): boolean {
  if (click.defaultPrevented || click.button !== 0) return false;
  if (click.metaKey || click.ctrlKey || click.shiftKey || click.altKey) return false;
  if (!a.href || a.download || a.noTransition) return false;
  if (a.target && a.target !== "_self") return false;
  return isRouteChange(a.href, loc);
}

/** The page may start a view transition: the API is there, the reader allows motion, the page is live. */
export function mayTransition(env: { hasApi: boolean; reduced: boolean; splash: boolean; hidden: boolean; online: boolean; saved: boolean }): boolean {
  return env.hasApi && !env.reduced && !env.splash && !env.hidden && env.online && !env.saved;
}
