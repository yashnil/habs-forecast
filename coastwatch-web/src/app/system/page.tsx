import { isChartMode } from "@/nx/ds/mode";
import { Catalog } from "./Catalog";

export default async function SystemPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const sp = await searchParams;
  const mode = typeof sp.mode === "string" && isChartMode(sp.mode) ? sp.mode : "day";
  return <Catalog initialMode={mode} />;
}
