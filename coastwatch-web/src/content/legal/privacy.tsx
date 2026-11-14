import Link from "next/link";
import { OWNER_CONTACT } from "./contact";

/** The sections, for the page's contents (the h2s below, in order). */
export const PRIVACY_CONTENTS = [
  { id: "collect", label: "What we collect" },
  { id: "use", label: "How we use it" },
  { id: "share", label: "Who we share it with" },
  { id: "retention", label: "How long we keep it" },
  { id: "choices", label: "Seeing, changing and deleting your information" },
  { id: "dnt", label: "Do Not Track" },
  { id: "children", label: "Children" },
  { id: "security", label: "Security" },
  { id: "changes", label: "Changes to this policy" },
  { id: "contact", label: "Contact" },
];

/** CalOPPA-oriented privacy policy for the Neritic pilot (was src/app/privacy/page.tsx), word for word. Draft until the owner and counsel review it. The page frame is src/nx/pages/LegalPage.tsx. */
export function PrivacyBody() {
  return (
    <>
    <p>
      Neritic is an independent research preview that summarises California coastal and fisheries information for fishermen. This policy explains what we collect when you use the Neritic website and its
      pilot features (saved ports, alerts, notifications, text alerts, feedback, trip logs and Ask Neritic), why, who it is shared with, how long we keep it, and how you can see or delete it. &ldquo;We&rdquo; means
      the people who run Neritic. Contact: <strong>{OWNER_CONTACT}</strong>.
    </p>
    <p>
      <strong>The short version:</strong> you can use Neritic without an account. Without one, your choices stay in your own browser and we never see them. If you create an account, we keep only what the pilot needs,
      we never sell your information, and we never share it for marketing.
    </p>

    <h2 id="collect">What we collect</h2>
    <h3>Kept only in your browser (no account)</h3>
    <ul>
      <li>Your home port, the port you opened last, your fisheries and vessel length band, and the ports you follow with what should alert you at each.</li>
      <li>Your alerts, and what each followed port&rsquo;s records said the last time this browser checked them. Alerts are made in your browser when you open Neritic; nothing checks your ports while it is closed, and no text, email or phone notification is sent.</li>
      <li>What Today showed at each port on your last visit, so it can say what changed since.</li>
      <li>Trips you save on My Coast: your vessels, costs, prices, catch and crew shares. These never leave your browser.</li>
      <li>Pages, records and chart pictures you save for offline use.</li>
      <li>Display choices (day, dusk or night chart; introductions you dismissed; where you placed, sized or collapsed the panels on a page).</li>
      <li>In the pilot account page: role, saved ports, feedback answers and trip feedback, and any confirmation or consent you gave there, with the time and the words shown.</li>
    </ul>
    <p>
      These stay on your device (browser storage and the browser&rsquo;s offline cache). We do not receive them. &ldquo;Delete everything on this device&rdquo; on the <Link href="/account">account page</Link> removes all of
      them, including saved trips and offline copies; so does clearing your browser&rsquo;s data for this site. The one exception is a question you type into Ask Neritic, which is sent to our server to be answered
      (see Ask Neritic logs below).
    </p>
    <h3>If you create an account</h3>
    <ul>
      <li><strong>Identifiers:</strong> your email address (to send sign-in codes) and an account id.</li>
      <li><strong>Profile:</strong> role, fisheries, home port, vessel length band, and your 13+ attestation with its date.</li>
      <li><strong>Preferences:</strong> saved ports and alert subscriptions (which changes, which channel).</li>
      <li><strong>Alerts:</strong> the alerts we created for you, when they were sent, and whether you read them.</li>
      <li><strong>Notifications:</strong> if you turn them on, your browser&rsquo;s push address and keys (no location, no contacts).</li>
      <li><strong>Text alerts:</strong> if you opt in, your mobile number and a record of your consent (time and exact wording).</li>
      <li><strong>Feedback and trip logs:</strong> &ldquo;Was this useful?&rdquo; answers and notes, and trip logs you choose to write (date, port, fishery, an outcome band, conditions you observed). We never ask for catch amounts or fishing locations.</li>
      <li><strong>Consent records:</strong> what you agreed to and when (terms, privacy, age, text alerts, notifications).</li>
    </ul>
    <h3>Collected automatically</h3>
    <ul>
      <li>Our hosting provider records standard server logs (IP address, browser type, page requested, time) to run and secure the service.</li>
      <li>We do not use advertising or cross-site tracking cookies. Signing in sets a cookie that keeps you signed in. Any product analytics we add will be first-party and cookieless.</li>
    </ul>
    <h3>Ocean data Neritic fetches</h3>
    <p>
      Neritic&rsquo;s server fetches the live ocean data a page shows (NOAA WCOFS model guidance, NOAA CoastWatch’s gap-filled chlorophyll estimate, NDBC buoys, CO-OPS tides and National Weather
      Service marine forecasts) directly from NOAA. Those requests name only the data and the area of the coast; they carry nothing about you, your ports or your trips, and your browser does not contact NOAA.
    </p>
    <h3>Ask Neritic</h3>
    <p>If you use Ask Neritic, your question is sent to our AI provider to produce an answer from Neritic&rsquo;s published data. We keep questions and answers only if you agree, for up to 90 days, to check accuracy.</p>

    <h2 id="use">How we use it</h2>
    <ul>
      <li>To provide the features you ask for: remembering your ports, building and delivering alerts, notifications and texts, and signing you in.</li>
      <li>To understand whether Neritic is useful and accurate (feedback and trip logs, reviewed in aggregate).</li>
      <li>To keep the service secure, prevent abuse, and meet legal obligations.</li>
    </ul>
    <p>We do not sell personal information, and we do not use it for targeted advertising.</p>

    <h2 id="share">Who we share it with</h2>
    <p>We share personal information only with service providers that run Neritic for us, under contracts that limit their use to that purpose:</p>
    <ul>
      <li>Supabase (database and sign-in, hosted in the US West region).</li>
      <li>Our website host (for example Vercel), which processes requests and server logs.</li>
      <li>Our email provider, to send sign-in codes.</li>
      <li>Browser push services (Apple, Google or Mozilla), which relay encrypted notifications to your device.</li>
      <li>Our SMS provider (Twilio), only if you opt in to text alerts and once they are live.</li>
      <li>Our AI provider (Anthropic), only for Ask Neritic questions.</li>
    </ul>
    <p>
      <strong>Mobile numbers and text-message opt-in data and consent are never shared with or sold to third parties or affiliates for marketing or promotional purposes.</strong> We may disclose information if
      the law requires it, or to protect someone&rsquo;s safety, and to a successor if Neritic is transferred (this policy would still apply).
    </p>

    <h2 id="retention">How long we keep it</h2>
    <ul>
      <li><strong>Browser-only data:</strong> until you delete it.</li>
      <li><strong>Account, profile, saved ports and alert subscriptions:</strong> until you delete your account.</li>
      <li><strong>Alert inbox and delivery records:</strong> 180 days, then deleted.</li>
      <li><strong>Feedback and trip logs:</strong> until you delete your account, or two years after your last sign-in, whichever is first.</li>
      <li><strong>Consent records:</strong> while your account exists.</li>
      <li><strong>Ask Neritic logs:</strong> 90 days, and only with your consent.</li>
      <li><strong>Server logs:</strong> up to 30 days, as kept by our host.</li>
      <li><strong>Backups:</strong> deleted data leaves our database backups within 30 days.</li>
    </ul>

    <h2 id="choices">Seeing, changing and deleting your information</h2>
    <ul>
      <li>The <Link href="/account">account page</Link> shows and lets you change everything you have given us.</li>
      <li>&ldquo;Delete my account&rdquo; on that page deletes your account and every record linked to it immediately (backups as above). &ldquo;Delete everything on this device&rdquo; removes everything Neritic keeps in your browser: ports, fisheries, follows, alerts, saved trips, visit history and offline copies. If you are signed in, it does not sign you out or delete your account.</li>
      <li>To stop texts, reply STOP. To stop notifications, turn them off on the account page or in your browser settings.</li>
      <li>You can also email <strong>{OWNER_CONTACT}</strong> to ask for a copy of your information, a correction, or deletion. We will answer within 30 days.</li>
      <li>California residents have rights under the California Consumer Privacy Act where it applies; we honour access and deletion requests from anyone, wherever they live.</li>
    </ul>

    <h2 id="dnt">Do Not Track</h2>
    <p>
      Some browsers send a &ldquo;Do Not Track&rdquo; signal. There is no common standard for it, so Neritic does not change its behaviour when it receives one; but Neritic does not track you across other
      websites, and we do not allow third parties to collect personal information about your activity on Neritic over time or across sites. We also honour Global Privacy Control signals as an opt-out of any
      sale or sharing (we do neither).
    </p>

    <h2 id="children">Children</h2>
    <p>Neritic is not directed to children under 13, and we do not knowingly collect personal information from them. Creating an account requires confirming you are 13 or older. If you believe a child under 13 has given us information, contact us and we will delete it.</p>

    <h2 id="security">Security</h2>
    <p>Accounts use one-time email codes (no passwords to steal). Each person&rsquo;s records are isolated in the database by row-level security, and data is encrypted in transit. No system is perfectly secure; we will notify affected people of a breach as the law requires.</p>

    <h2 id="changes">Changes to this policy</h2>
    <p>We will post any change here with a new effective date. If a change is material, we will tell account holders by email or in the app before it takes effect.</p>

    <h2 id="contact">Contact</h2>
    <p>
      Questions or requests: <strong>{OWNER_CONTACT}</strong>.
    </p>
    </>
  );
}
