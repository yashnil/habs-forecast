import { NotFoundBody } from "@/nx/pages/errors";
import { StandaloneShell } from "@/nx/pages/StandaloneShell";
import { shellPorts } from "@/nx/shell/ports.server";

export const metadata = { title: "Not found" };

/** Any address no route claims, in the same shell (bar, menu, tab bar) without the chart: src/nx/pages/StandaloneShell. */
export default async function RootNotFound() {
  return (
    <StandaloneShell ports={await shellPorts()}>
      <NotFoundBody />
    </StandaloneShell>
  );
}
