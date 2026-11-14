import { TermsBody, TERMS_CONTENTS } from "@/content/legal/terms";
import { LegalPage } from "@/nx/pages/LegalPage";
import { registryTrust } from "@/nx/pages/registry.server";

export const dynamic = "force-dynamic";
export const metadata = { title: "Terms of Use" };

/** Terms of Use for the Neritic pilot, in the shell (src/nx/pages). Draft until the owner and counsel review it; the words are in src/content/legal. */
export default async function Page() {
  return (
    <LegalPage id="terms" title="Terms of Use" testId="terms-page" trust={await registryTrust()} contents={TERMS_CONTENTS}>
      <TermsBody />
    </LegalPage>
  );
}
