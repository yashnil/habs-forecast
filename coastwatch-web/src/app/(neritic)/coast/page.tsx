import { permanentRedirect } from "next/navigation";
import { coastRedirect } from "@/nx/shell/redirects";
import { CoastMargin } from "@/nx/shell/views";

export const metadata = { title: "My Coast" };

/** My Coast: gates, the sea window, trip economics. Placeholder margin until the My Coast feature lands. */
export default async function CoastPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const to = coastRedirect(await searchParams);
  if (to) permanentRedirect(to);
  return <CoastMargin />;
}
