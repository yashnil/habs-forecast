import { notFound } from "next/navigation";
import { AccountApp } from "@/pilot/account/AccountApp";
import { pilotAccountConfig } from "@/pilot/account/config";
import { loadPortTodays } from "@/pilot/alerts/source";
import { PilotShell } from "@/pilot/shell/PilotShell";
import { loadTrust } from "@/lib/data";

export const dynamic = "force-dynamic";
export const metadata = { title: "Pilot account", robots: { index: false } };

/**
 * /account/pilot: the private pilot's account page (email sign-in, synced ports, trip feedback), in the pilot's
 * shell. It exists only where accounts are connected (Supabase env): without them there is nothing to sign in
 * to, the address is not found, and nothing in the v13 shell links here. What a reader keeps on the device
 * without an account is `/account` (src/nx/account) and `/alerts` (src/nx/alerts).
 */
export default async function AccountPage() {
  const config = pilotAccountConfig();
  if (!config.supabase) notFound();
  const facts = await loadPortTodays();
  return (
    <PilotShell testid="account-shell" account={config} trust={await loadTrust()}>
      <AccountApp config={config} todays={facts.todays} sample={facts.sample} />
    </PilotShell>
  );
}
