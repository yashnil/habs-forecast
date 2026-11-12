import { AccountPage } from "@/nx/account/AccountPage";
import { readiness } from "@/pilot/readiness";

export const dynamic = "force-dynamic";
export const metadata = { title: "Your data and preferences", robots: { index: false } };

/**
 * `/account`: what Neritic keeps in this browser and how it behaves (src/nx/account). Everything on it is read
 * from the device after hydration. Accounts are decided from configuration at request time: where they are not
 * connected (this preview) the page offers no sign-in at all; where they are, it links to the pilot's page.
 */
export default function Account() {
  return <AccountPage pilotAccount={readiness("accounts") !== "not_configured"} />;
}
