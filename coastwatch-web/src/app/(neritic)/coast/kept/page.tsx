import { KeptBriefing } from "@/nx/pages/KeptBriefing";

export const metadata = { title: "Kept briefings" };

/**
 * Briefings kept on this device: dated copies to read or print. Read from the device only (no data request), and
 * marked as a kept copy so a saved copy is never mistaken for current status. v15.1: in the shell (src/nx/pages).
 */
export default function KeptPage() {
  return <KeptBriefing />;
}
