import { KeptView } from "@/components/coast/KeptView";

export const metadata = { title: "Kept briefings" };

/**
 * Briefings kept on this device: dated copies to read or print. Read from the device only (no data
 * request), and shown apart from the live site so a saved copy is never mistaken for current status.
 */
export default function KeptPage() {
  return <KeptView />;
}
