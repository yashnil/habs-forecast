"use client";

import { runCheck } from "./check";
import { MAX_FOLLOWS, follow, unfollow, type FollowResult } from "./follow";
import { useFollows } from "./useInbox";

/**
 * One port's follow state, shared by every page (Today's ticket, Alerts, the chart's ultramarine port mark): all read
 * the same device record (nx-follow-v1), so following on one page shows at once on the others and in other tabs.
 *
 *   const f = useFollow(port.key);
 *   <FollowButton name={port.name} following={!!f.following} onChange={f.set} disabled={f.full && !f.following} />
 *
 * `following` is undefined until the device's storage is read (server HTML and first render agree). Following a
 * port notes its records as they are now (a check, forced), so a change from here on becomes an alert.
 */
export function useFollow(port: string | null | undefined): {
  following: boolean | undefined;
  /** when it was followed (device clock), "" for older records */
  since: string | null;
  /** this device follows the most ports it keeps */
  full: boolean;
  set: (next: boolean) => FollowResult | { ok: true; persisted: boolean };
} {
  const follows = useFollows();
  const mine = port ? follows?.find((f) => f.port === port) : undefined;
  return {
    following: follows === undefined ? undefined : !!mine,
    since: mine?.since ?? null,
    full: (follows?.length ?? 0) >= MAX_FOLLOWS,
    set: (next) => {
      if (!port) return { ok: false, why: "not_a_port" } as const;
      if (!next) return { ok: true, persisted: unfollow(port) };
      const r = follow(port);
      if (r.ok) void runCheck({ force: true });
      return r;
    },
  };
}
