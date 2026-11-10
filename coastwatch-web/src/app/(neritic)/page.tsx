import { permanentRedirect } from "next/navigation";
import { rootRedirect } from "@/nx/shell/redirects";
import { RootEntry } from "@/nx/shell/views";

type SP = Promise<Record<string, string | string[] | undefined>>;

/**
 * `/`, the one front door (journeys.md): a first visit gets the opening, which becomes home in place; a
 * returning reader with a saved port gets Today here, with no redirect. Old Explore links (`/?view=map`,
 * `/?sel=…`) go to /explore with their parameters.
 */
export default async function Home({ searchParams }: { searchParams: SP }) {
  const to = rootRedirect(await searchParams);
  if (to) permanentRedirect(to);
  return <RootEntry />;
}
