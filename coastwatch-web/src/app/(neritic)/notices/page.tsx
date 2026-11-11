import { loadData } from "@/lib/data";
import { isPortKey } from "@/lib/portRecord";
import { buildNotices } from "@/nx/notices/model";
import { NoticesPage } from "@/nx/notices/NoticesPage";
import { shellPorts } from "@/nx/shell/ports.server";

export const metadata = { title: "Official notices" };

type SP = Promise<Record<string, string | string[] | undefined>>;

/**
 * Every current official notice for a port (`?place=`) and the rest of the coast, from the published registry.
 * Reached from Today's Notices and Others rows and from Alerts; Today's tab stays current (Shell activeDestination).
 * One segment deep like the four destinations, so the shared layout (and the one chart) persists into it.
 */
export default async function Notices({ searchParams }: { searchParams: SP }) {
  const raw = (await searchParams).place;
  const key = typeof raw === "string" && isPortKey(raw) ? raw : null;
  const [d, ports] = await Promise.all([loadData(), shellPorts()]);
  const port = key ? (ports.find((p) => p.key === key) ?? null) : null;
  const now = (process.env.CW_FIXED_NOW ? new Date(process.env.CW_FIXED_NOW) : new Date()).toISOString();
  const view = d.ok ? buildNotices(d.official, port, new Date(now)) : null;
  return <NoticesPage view={view} now={now} />;
}
