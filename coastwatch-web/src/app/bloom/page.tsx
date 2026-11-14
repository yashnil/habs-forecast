import { permanentRedirect } from "next/navigation";
import { bloomRedirect } from "@/nx/shell/redirects";

/** The v2 station page is Research now (journeys.md: `/bloom` → `/research`), with its parameters carried. */
export default async function BloomPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  permanentRedirect(bloomRedirect(await searchParams));
}
