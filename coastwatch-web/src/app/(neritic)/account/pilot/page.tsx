import { notFound } from "next/navigation";
import { AccountApp } from "@/pilot/account/AccountApp";
import { pilotAccountConfig } from "@/pilot/account/config";
import { loadPortTodays } from "@/pilot/alerts/source";
import { InfoPage } from "@/nx/pages/InfoPage";
import { PilotAccountFrame } from "@/nx/pages/PilotAccount";
import { registryTrust } from "@/nx/pages/registry.server";
import "@/nx/pages/bridge.css";

export const dynamic = "force-dynamic";
export const metadata = { title: "Pilot account", robots: { index: false } };

/**
 * /account/pilot: the private pilot's account page (email sign-in, synced ports, trip feedback). It exists only where
 * accounts are connected (Supabase env): without them there is nothing to sign in to, the address is not found, and
 * nothing in the shell links here. What a reader keeps on the device without an account is `/account` (src/nx/account)
 * and `/alerts` (src/nx/alerts). v15.1: in the shell (src/nx/pages); the pilot's form takes the design system's
 * colours and type through the token bridge.
 */
export default async function AccountPage() {
  const config = pilotAccountConfig();
  if (!config.supabase) notFound();
  const [facts, trust] = await Promise.all([loadPortTodays(), registryTrust()]);
  return (
    <InfoPage id="account-pilot" testId="account-shell" trust={trust} width="wide">
      <div className="nxp-bridge">
        <PilotAccountFrame config={config}>
          <AccountApp config={config} todays={facts.todays} sample={facts.sample} />
        </PilotAccountFrame>
      </div>
    </InfoPage>
  );
}
