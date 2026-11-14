import { SmsTermsBody, SMS_TERMS_CONTENTS } from "@/content/legal/sms-terms";
import { LegalPage } from "@/nx/pages/LegalPage";
import { registryTrust } from "@/nx/pages/registry.server";

export const dynamic = "force-dynamic";
export const metadata = { title: "SMS Terms" };

/** SMS Terms for the Neritic pilot, in the shell (src/nx/pages). Draft until the owner and counsel review it; the words are in src/content/legal. */
export default async function Page() {
  return (
    <LegalPage id="sms-terms" title="SMS Terms" testId="sms-terms-page" trust={await registryTrust()} contents={SMS_TERMS_CONTENTS}>
      <SmsTermsBody />
    </LegalPage>
  );
}
