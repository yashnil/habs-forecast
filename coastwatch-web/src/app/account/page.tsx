import { AppShell } from "@/components/AppShell";
import { loadData } from "@/lib/data";
import { AccountApp } from "@/pilot/account/AccountApp";
import { pilotAccountConfig } from "@/pilot/account/config";
import { loadPortTodays } from "@/pilot/alerts/source";

export const dynamic = "force-dynamic";
export const metadata = { title: "Your ports and alerts", robots: { index: false } };

/**
 * /account (pilot): saved ports, fisheries, alert preferences and the alert inbox. Readiness is
 * decided here from env at request time; with no Supabase credentials everything stays on the device.
 */
export default async function AccountPage() {
  const [data, facts] = await Promise.all([loadData(), loadPortTodays()]);
  return (
    <AppShell active="about" manifest={data.ok ? data.manifest : null} official={data.ok ? data.official : null}>
      <AccountApp config={pilotAccountConfig()} todays={facts.todays} stubFacts={facts.stub} />
    </AppShell>
  );
}
