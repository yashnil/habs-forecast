import Link from "next/link";
import { AppShell } from "@/components/AppShell";
import { loadData } from "@/lib/data";
import { LegalPage, OWNER_CONTACT } from "@/pilot/account/LegalPage";

export const dynamic = "force-dynamic";
export const metadata = { title: "Terms of Use" };

/** Terms of use for the Neritic pilot. Draft until the owner and counsel review them. */
export default async function TermsPage() {
  const data = await loadData();
  return (
    <AppShell active="about" manifest={data.ok ? data.manifest : null} official={data.ok ? data.official : null}>
      <LegalPage title="Terms of Use" testId="terms-page">
        <p>
          These terms apply when you use the Neritic website and its pilot features. By using Neritic you agree to them. If you do not agree, please do not use Neritic. Contact: <strong>{OWNER_CONTACT}</strong>.
        </p>

        <h2 id="what">What Neritic is, and is not</h2>
        <ul>
          <li>
            <strong>Not an official source.</strong> Neritic is an independent research preview. It is not affiliated with or endorsed by NOAA, the California Department of Fish and Wildlife (CDFW), the California
            Department of Public Health (CDPH), the National Weather Service (NWS) or the US Coast Guard (USCG). The official sources are: <strong>CDFW</strong> for fishing seasons, closures and whale-entanglement
            (RAMP) actions; <strong>CDPH</strong> for seafood-safety advisories and quarantines; <strong>NWS</strong> for marine forecasts and warnings; and <strong>USCG</strong> for navigation safety and bar
            conditions. Check them before you act. Neritic&rsquo;s copies of official notices may be late, incomplete or wrong, and are marked &ldquo;Not verified&rdquo; until a person has checked them.
          </li>
          <li>
            <strong>Not for navigation.</strong> Do not use Neritic to navigate, to decide whether it is safe to go to sea or cross a bar, or in an emergency. In an emergency call the Coast Guard on VHF channel 16 or
            dial 911.
          </li>
          <li>
            <strong>Not a seafood-safety service.</strong> Satellite chlorophyll is not toxin: it shows algae, and it cannot tell you whether seafood can be eaten. Model forecasts (such as NOAA C-HARM bloom probabilities and
            Neritic&rsquo;s chlorophyll outlook) are estimates, and observations show the past, not the future. Only CDPH advisories and laboratory tests say whether seafood is safe.
          </li>
          <li>
            <strong>AI answers may be wrong.</strong> Ask Neritic uses an AI model to answer from Neritic&rsquo;s published data. It can misread data or be out of date. Check its cited sources, and never rely on it for
            safety, legal or navigation decisions.
          </li>
          <li>
            <strong>Alerts can be late or missed.</strong> Alerts, notifications and texts depend on data feeds, networks and devices outside our control. Not receiving an alert does not mean nothing changed.
          </li>
        </ul>

        <h2 id="responsibility">Your decisions are yours</h2>
        <p>You are responsible for your decisions about fishing, going to sea, harvesting and eating seafood, and for following the law and official orders. Neritic is information to help you look in the right places, not advice or a substitute for official sources or your own judgment.</p>

        <h2 id="accounts">Accounts</h2>
        <ul>
          <li>You must be 13 or older to create an account. If you are under 18, use Neritic with a parent or guardian&rsquo;s permission.</li>
          <li>Keep access to your email secure: anyone who can read it can sign in with a code. Tell us if you think someone else has used your account.</li>
          <li>You can delete your account at any time on the <Link href="/account">account page</Link>. We may suspend accounts that abuse the service.</li>
        </ul>

        <h2 id="use">Acceptable use</h2>
        <p>Do not misuse Neritic: no attempts to break or overload it, to access other people&rsquo;s data, to scrape it in a way that harms the service, or to present Neritic&rsquo;s information as an official notice. Neritic&rsquo;s data comes from public sources whose own terms also apply.</p>

        <h2 id="pilot">A pilot, and it may change</h2>
        <p>Neritic is free, experimental and may change, pause or stop at any time, with or without notice. Features labelled &ldquo;Pilot demo&rdquo; or &ldquo;Not yet connected&rdquo; may not work.</p>

        <h2 id="warranty">No warranty</h2>
        <p>
          NERITIC IS PROVIDED &ldquo;AS IS&rdquo; AND &ldquo;AS AVAILABLE&rdquo;, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING ACCURACY, COMPLETENESS, TIMELINESS, MERCHANTABILITY, FITNESS FOR A
          PARTICULAR PURPOSE AND NON-INFRINGEMENT. WE DO NOT WARRANT THAT IT WILL BE UNINTERRUPTED, ERROR-FREE OR SECURE.
        </p>

        <h2 id="liability">Limitation of liability</h2>
        <p>
          TO THE FULLEST EXTENT THE LAW ALLOWS, THE PEOPLE WHO RUN NERITIC ARE NOT LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL OR PUNITIVE DAMAGES, OR FOR LOST CATCH, PROFITS, GEAR, OR TRIPS,
          ARISING FROM YOUR USE OF OR RELIANCE ON NERITIC, EVEN IF WE WERE TOLD THEY WERE POSSIBLE. OUR TOTAL LIABILITY FOR ANY CLAIM IS LIMITED TO US$100. SOME JURISDICTIONS DO NOT ALLOW THESE LIMITS, SO THEY
          MAY NOT APPLY TO YOU.
        </p>

        <h2 id="law">Governing law</h2>
        <p>These terms are governed by the laws of the State of California, without regard to its conflict-of-laws rules. Disputes will be heard in the state or federal courts located in California, unless the law requires otherwise.</p>

        <h2 id="changes">Changes</h2>
        <p>We may update these terms. We will post changes here with a new effective date, and tell account holders about material changes before they take effect. Continuing to use Neritic after that means you accept them.</p>

        <h2 id="privacy">Privacy and texts</h2>
        <p>
          The <Link href="/privacy">Privacy Policy</Link> explains how we handle your information, and the <Link href="/sms-terms">SMS Terms</Link> apply to text alerts.
        </p>
      </LegalPage>
    </AppShell>
  );
}
