import { isPortKey } from "@/lib/portRecord";
import { loadToday } from "@/pilot/today/load";
import { TodayPage } from "@/nx/today/TodayPage";

export const metadata = { title: "Today at your port" };

type SP = Promise<Record<string, string | string[] | undefined>>;

/**
 * Today: the season status, the next date and what changed at a port. With `?place=` the port's published
 * records arrive with the page; without it the saved port is read on the device and its records fetched.
 */
export default async function Today({ searchParams }: { searchParams: SP }) {
  const raw = (await searchParams).place;
  const place = typeof raw === "string" && isPortKey(raw) ? raw : null;
  const l = place ? await loadToday(place).catch(() => null) : null;
  const initial = place && l?.ok && l.inputs ? { key: place, inputs: l.inputs, fixture: l.fixture } : null;
  return <TodayPage initial={initial} />;
}
