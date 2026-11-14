/**
 * The pilot's front door (docs/coastwatch/pilot/README.md): with NERITIC_PILOT=1 a bare `/` opens
 * Today at your port. Any link that names what to open keeps opening Explore: `/?region=…`,
 * `?port=`, `?var=`, `?lead=`, `?layer=`, `?inspect=`, `?sel=`, `?at=`, and every other parameter
 * Explore reads. Only a bare address, or one carrying nothing but a campaign tag, is redirected.
 * Explore itself stays at a stable address under the flag: `/?view=map` (the nav's Explore link).
 *
 * Without the flag nothing changes: `/` is Explore, as on the public release.
 */
// v12: one front door (src/app/(neritic)/page.tsx). Explore has its own address whatever the flag.
export const EXPLORE_URL = "/explore";

/** Read on the server at request time (the root layout hands it to the top bar). */
export function pilotFrontDoor(env: Record<string, string | undefined> = process.env): boolean {
  return env.NERITIC_PILOT === "1";
}

/** Parameters that say nothing about what to open (a shared link's campaign tags). */
const IGNORED = /^(utm_[a-z_]+|fbclid|gclid|msclkid|mc_[a-z]+|ref|source)$/i;

/** Does this address ask for Explore (a deep link), or is it a bare visit to the site? */
export function namesExplore(searchParams: Record<string, string | string[] | undefined>): boolean {
  return Object.entries(searchParams).some(([k, v]) => v !== undefined && !IGNORED.test(k));
}

/** Where `/` goes: "/today" for a bare visit under the pilot flag, otherwise null (render Explore). */
export function frontDoorRedirect(searchParams: Record<string, string | string[] | undefined>, env: Record<string, string | undefined> = process.env): string | null {
  return pilotFrontDoor(env) && !namesExplore(searchParams) ? "/today" : null;
}

/** The Explore link for the nav and for "open the map" links. */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export const exploreHref = (_pilot: boolean) => EXPLORE_URL;
