import Link from "next/link";
import { PROGRAM_NAME, SMS_CONSENT_TEXT } from "@/pilot/account/consent";
import { LegalPage, OWNER_CONTACT } from "@/pilot/account/LegalPage";

export const dynamic = "force-dynamic";
export const metadata = { title: "SMS Terms" };

/** SMS program terms (for 10DLC registration and for people who opt in). Draft until reviewed. */
export default async function SmsTermsPage() {
  return (
      <LegalPage title="SMS Terms" testId="sms-terms-page">
        <p role="note" className="rounded-lg border border-n-line-strong px-3 py-2 text-[14.5px]" data-testid="sms-status">
          Status: text alerts are not being sent yet. The program is awaiting carrier registration; until then, only the pilot team&rsquo;s verified test phones receive messages.
        </p>

        <h2 id="program">Program</h2>
        <p>
          <strong>Program name:</strong> {PROGRAM_NAME}. <strong>Sender:</strong> Neritic, an independent research preview for California coastal information. Contact: <strong>{OWNER_CONTACT}</strong>.
        </p>

        <h2 id="messages">What we send, and how often</h2>
        <ul>
          <li>Alerts about the ports and changes you choose: official fishing status changes (CDFW, CDPH), new domoic acid test results, whale-entanglement (RAMP) actions, sharp rises in NOAA C-HARM bloom-risk forecasts, and stale-data warnings.</li>
          <li>One confirmation text when you opt in, and replies to HELP and STOP.</li>
          <li>If you use Ask Neritic by text (when available), answers to the questions you send.</li>
          <li><strong>Frequency varies</strong> with official changes, up to 5 messages per week. No marketing messages.</li>
        </ul>
        <p>
          <strong>Msg &amp; data rates may apply.</strong> Carriers are not liable for delayed or undelivered messages.
        </p>

        <h2 id="optin">How you opt in</h2>
        <p>On the <Link href="/account">account page</Link>, you tick an unticked box next to this text and enter your mobile number:</p>
        <blockquote className="border-l-2 border-n-line-strong pl-4 italic" data-testid="sms-terms-disclosure">
          {SMS_CONSENT_TEXT}
        </blockquote>
        <p>
          <strong>Consent is not a condition</strong> of using Neritic or of any purchase. We record the time you opted in and the exact words shown.
        </p>

        <h2 id="stop">STOP and HELP</h2>
        <ul>
          <li>
            Reply <strong>STOP</strong> (or END, CANCEL, UNSUBSCRIBE, QUIT) at any time to stop all messages. We will send one text confirming you are unsubscribed, and nothing after that.
          </li>
          <li>
            Reply <strong>HELP</strong> for help, or email <strong>{OWNER_CONTACT}</strong>.
          </li>
          <li>You can also turn text alerts off on the account page.</li>
        </ul>

        <h2 id="privacy">Privacy</h2>
        <p>
          <strong>Mobile numbers and text-message opt-in data and consent are never shared with or sold to third parties or affiliates for marketing or promotional purposes.</strong> We use your number only to send
          the messages described here, through our SMS provider. See the <Link href="/privacy">Privacy Policy</Link>.
        </p>

        <h2 id="limits">Not an official or emergency service</h2>
        <p>
          Texts can be late or not arrive. They are not official notices and not safety advice: CDFW, CDPH, NWS and the US Coast Guard are the official sources. In an emergency call the Coast Guard on VHF channel
          16 or dial 911. The <Link href="/terms">Terms of Use</Link> also apply.
        </p>
      </LegalPage>
  );
}
