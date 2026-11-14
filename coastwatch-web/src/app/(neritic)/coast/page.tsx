import { permanentRedirect } from "next/navigation";
import { coastRedirect } from "@/nx/shell/redirects";
import { CoastPage } from "@/nx/coast/CoastPage";

export const metadata = { title: "My Coast" };

/** My Coast: plan a trip. The gates for your fishery, the sea window, then a worksheet from your own numbers. */
export default async function Coast({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const to = coastRedirect(await searchParams);
  if (to) permanentRedirect(to);
  return <CoastPage />;
}
