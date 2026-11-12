import { readJSON, writeJSON } from "@/nx/personal/store";

/**
 * The ports where the reader put the "Follow this port" line away (localStorage "nx-follow-hint-v1"): a list
 * of port keys, newest first. A dismissed line never comes back at that port.
 */
export const FOLLOW_HINT_KEY = "nx-follow-hint-v1";
const MAX_DISMISSED = 60;

const isList = (v: unknown): v is string[] => Array.isArray(v) && v.every((x) => typeof x === "string");

export const dismissedPorts = () => readJSON<string[]>(FOLLOW_HINT_KEY, [], isList);

export function dismissFollowHint(port: string) {
  const cur = dismissedPorts().filter((p) => p !== port);
  writeJSON(FOLLOW_HINT_KEY, [port, ...cur].slice(0, MAX_DISMISSED));
}
