import { loadExploreModel } from "@/nx/explore/load.server";
import { Explore, ExploreUnavailable } from "@/nx/explore/Explore";

export const metadata = { title: "Explore" };

/** Explore: observations and models in their place and time, on the full sheet (journeys.md Journey 3). */
export default async function ExplorePage() {
  const l = await loadExploreModel();
  if (!l.ok) return <ExploreUnavailable why={l.error} />;
  return <Explore model={l.model} />;
}
