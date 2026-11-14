"use client";

import Link from "next/link";
import { useSyncExternalStore } from "react";
import { Button, Hint } from "@/nx/ds";
import { readRaw, subscribe } from "@/nx/personal/store";
import { runCheck } from "./check";
import { MAX_FOLLOWS, follow } from "./follow";
import { FOLLOW_HINT_KEY, dismissFollowHint, dismissedPorts } from "./followHint";
import { useFollows } from "./useInbox";
import { DELIVERY_SHORT } from "./words";
import s from "./follow.module.css";

/**
 * The one line of first-use guidance for alerts, for Today (the slot under the status and its dates):
 *
 *   not followed   "Follow Moss Landing …" with one button, and a close control that puts it away for good at
 *                  this port (it is never a tutorial and never comes back once dismissed);
 *   followed       a quiet line saying so, when the port is checked, and where the alerts are.
 *
 * It renders nothing until the device's storage has been read, so the server's HTML and the first client
 * render agree, and nothing at all when the reader dismissed it.
 *
 *   import { FollowLine } from "@/nx/alerts/FollowLine";
 *   <FollowLine port={{ key: port.key, name: port.name }} />
 */
export function FollowLine({ port, className }: { port: { key: string; name: string }; className?: string }) {
  const follows = useFollows();
  const raw = useSyncExternalStore(
    subscribe,
    () => readRaw(FOLLOW_HINT_KEY) ?? "",
    () => undefined,
  );
  if (follows === undefined || raw === undefined) return null;
  const href = `/alerts?place=${encodeURIComponent(port.key)}`;
  if (follows.some((f) => f.port === port.key))
    return (
      <p className={[s.following, className].filter(Boolean).join(" ")} data-testid="nx-follow-line" data-state="following">
        Following {port.name}. {DELIVERY_SHORT}{" "}
        <Link href={href} className={s.link}>
          Alerts
        </Link>
      </p>
    );
  if (dismissedPorts().includes(port.key)) return null;
  const full = follows.length >= MAX_FOLLOWS;
  return (
    <div className={className} data-testid="nx-follow-line" data-state="offer">
      <Hint onDismiss={() => dismissFollowHint(port.key)}>
        <span className={s.offer}>
          <span>
            Follow {port.name} and Neritic lists what changed here (season status, notices, tests, whale measures) each time you open it. {DELIVERY_SHORT}
          </span>
          {full ? (
            <Link href={href} className={s.link}>
              You follow {MAX_FOLLOWS} ports, the most this device keeps. Manage them in Alerts
            </Link>
          ) : (
            <Button
              variant="secondary"
              onClick={() => {
                if (follow(port.key).ok) void runCheck({ force: true });
              }}
              data-testid="nx-follow"
            >
              Follow {port.name}
            </Button>
          )}
        </span>
      </Hint>
    </div>
  );
}
