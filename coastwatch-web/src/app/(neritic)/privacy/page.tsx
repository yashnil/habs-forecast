import { PrivacyBody, PRIVACY_CONTENTS } from "@/content/legal/privacy";
import { LegalPage } from "@/nx/pages/LegalPage";
import { registryTrust } from "@/nx/pages/registry.server";

export const dynamic = "force-dynamic";
export const metadata = { title: "Privacy Policy" };

/** Privacy Policy for the Neritic pilot, in the shell (src/nx/pages). Draft until the owner and counsel review it; the words are in src/content/legal. */
export default async function Page() {
  return (
    <LegalPage id="privacy" title="Privacy Policy" testId="privacy-page" trust={await registryTrust()} contents={PRIVACY_CONTENTS}>
      <PrivacyBody />
    </LegalPage>
  );
}
