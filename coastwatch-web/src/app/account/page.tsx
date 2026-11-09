import { AccountApp } from "@/pilot/account/AccountApp";
import { pilotAccountConfig } from "@/pilot/account/config";
import { loadPortTodays } from "@/pilot/alerts/source";
import { PilotShell } from "@/pilot/shell/PilotShell";

export const dynamic = "force-dynamic";
export const metadata = { title: "Your ports and alerts", robots: { index: false } };

/**
 * /account (pilot): saved ports, fisheries, alert preferences and the alert inbox, in the pilot's
 * shell. Readiness is decided here from env at request time; with no Supabase credentials everything
 * stays on the device. The facts alerts compare are the ones Today shows (alerts/source.ts).
 * `?place=port-592` (Today's "Set an alert") puts that port first.
 */
export default async function AccountPage() {
  const facts = await loadPortTodays();
  return (
    <PilotShell testid="account-shell">
      <AccountApp config={pilotAccountConfig()} todays={facts.todays} sample={facts.sample} />
    </PilotShell>
  );
}
