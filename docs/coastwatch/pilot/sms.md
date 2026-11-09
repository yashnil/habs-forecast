# Ask Neritic by SMS: what the owner submits (A2P 10DLC, sole proprietor)

Status on 2026-11-08: the SMS code is built and tested (`/api/sms`), but nothing is registered, no Twilio account is connected, and no message has been sent. Until the campaign is approved, `/ask` shows an **"SMS in carrier review — preview only"** panel with the exact reply a phone would get.

US carriers block application-to-person texts from unregistered 10-digit numbers. A **sole-proprietor** A2P 10DLC registration is the right size for a pilot run by one person without an EIN. It takes about 1–3 weeks, so submit it as soon as the legal pages are live.

## Before you start

You need:

- **Public URLs from Track A**: the privacy policy (`https://<pilot-host>/privacy`) and the SMS terms (`https://<pilot-host>/sms-terms`). Reviewers open them. They must be live (not behind a login) before you submit.
- The privacy policy must include these two sentences, word for word or very close. Carriers reject campaigns without them:
  > No mobile information will be shared with third parties or affiliates for marketing or promotional purposes. Text messaging opt-in data and consent will not be shared with any third parties.
- A US mobile number you control. Twilio texts it a one-time code during brand registration, and you must reply within 24 hours.
- Your legal name, home or business address and email. A sole-proprietor brand has no EIN.
- Spending approval. These are approximate; confirm on Twilio's pricing page when you submit:
  - about $21 up front (brand registration plus campaign vetting);
  - about $2 per month for the campaign;
  - about $1.15 per month for the phone number;
  - per-message fees.
- Limits of a sole-proprietor registration:
  - one campaign and one phone number;
  - low throughput (about 1 message per second, roughly 1,000 segments a day to T-Mobile).

  That is plenty for a pilot.

## Steps in the Twilio Console

1. **Upgrade** the account from trial, after approving the spend. Trial accounts text only verified numbers and add a "Sent from your Twilio trial account" prefix, which can push replies to 3 segments.
2. Go to **Messaging → Regulatory Compliance → A2P 10DLC** and choose **Sole Proprietor**.
3. **Customer profile:** enter your name, address and email. Use the same details everywhere.
4. **Brand registration:**
   - Enter the mobile number for verification.
   - Reply **YES** to the text Twilio sends.
   - Brand name: "Neritic" (the public name in messages is "Neritic Alerts").
5. **Buy one local number**, a California area code, for example 831 for Monterey Bay.
6. **Create a Messaging Service** called "Neritic Alerts" and add the number to it.
   - Integration: "Send a webhook".
   - Request URL: `https://<pilot-host>/api/sms` (HTTP POST).
7. **Campaign:** the use case is fixed to "Sole Proprietor". Paste the text from the next section.
8. **Opt-out handling:** leave Twilio's **Advanced Opt-Out** on (default keywords STOP/START/HELP). Then set `NERITIC_SMS_TWILIO_OPTOUT=1` so Neritic records opt-outs but does not send a second reply.
9. When the campaign is **approved**, set these in the pilot deployment's environment and redeploy (deploys need the owner's OK):
   - `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN`;
   - `NERITIC_SMS_WEBHOOK_URL=https://<pilot-host>/api/sms` (the exact URL Twilio signs);
   - `NERITIC_SMS_MODE=live`.

   Without `NERITIC_SMS_MODE=live` the readiness label stays "Pilot demo".

## Campaign text to paste

**Campaign description**
> Neritic Alerts answers questions from commercial fishermen in Monterey Bay and Half Moon Bay about fishery status, crab toxin test results, sea and tide forecasts, bloom outlooks and whale-entanglement rules. All answers come from public CDFW, CDPH, NOAA and NWS data. Users text a question (for example "MOSS crab status") and receive one reply with its source and date. Users who opt in on the website may also receive alerts they choose when an official status or test result changes. Not an official source. Low volume, private pilot.

**Message flow / call to action** (how people opt in)
> Users opt in in one of two ways. (1) On the website (https://<pilot-host>/account), a signed-in user enters their mobile number and ticks an unchecked box labelled with the opt-in text below. Consent is not required to use the site. (2) A user texts a question or START to the Neritic number; the first reply confirms the subscription and gives HELP/STOP instructions. SMS terms: https://<pilot-host>/sms-terms. Privacy policy: https://<pilot-host>/privacy.

**Opt-in text** (beside the unchecked checkbox on the account page; Track A renders it)
> Text me Neritic Alerts at this number: answers to my texts and the alerts I choose. Message frequency varies. Msg & data rates may apply. Reply HELP for help, STOP to cancel. Consent is not a condition of purchase. See SMS Terms and Privacy Policy.

**Sample messages** (these are real outputs of the pilot code on its fixture data, trimmed to fit)

1. `Commercial Dungeness crab is delayed until at least 2027-01-01 because of humpback whale entanglements, not domoic acid. Applies to Zones 1-5, Oregon border to Point Conception. [Src: CDFW 11/5, unverified]`
2. `Dungeness crab from Monterey Bay (BMA G) tested below the action level on 2026-09-28. Result: average 5.5 ppm domoic acid in viscera, highest sample 9 ppm; action level 30 ppm. [Src: CDPH crab 9/28]`
3. `Tomorrow (Mon): NW wind 10 to 15 kt, seas 5 to 7 ft at 10 seconds (NWS zone PZZ535). Tide tomorrow at Monterey: high 5.4 ft at 7:12 AM, low 0.9 ft at 1:48 PM. [Src: NWS marine 11/8]`
4. `Neritic can't say whether seafood can be eaten. CDPH decides: cdph.ca.gov (Shellfish Advisories); closures: wildlife.ca.gov.`

**Keyword replies** (from `src/pilot/ask/sms.ts`, `SMS_TEXT`)

| Keyword | Reply |
|---|---|
| HELP, INFO | `Neritic Alerts: answers about Monterey Bay & Half Moon Bay fisheries from public data. Not an official source; CDFW/CDPH decide. Text e.g. "MOSS crab status". Msg&data rates may apply. Reply STOP to cancel.` |
| STOP, STOPALL, UNSUBSCRIBE, CANCEL, END, QUIT, OPTOUT, REVOKE | `Neritic Alerts: you're unsubscribed and won't get more messages. Reply START to resubscribe.` |
| START, UNSTOP, YES | `Neritic Alerts: you're subscribed. Ask e.g. "MOSS crab status" or set a home port: "HOME MOSS". Not an official source. Msg&data rates may apply. Reply HELP for help, STOP to cancel.` |

**Campaign attributes:**

- Embedded links: **yes**. The safety replies name cdph.ca.gov and wildlife.ca.gov. Use no URL shorteners.
- Embedded phone numbers: **no**.
- Age-gated content: **no**.
- Direct lending: **no**.

## How the webhook behaves (for reviewers and testing)

- **Signature:** every request's `X-Twilio-Signature` is checked against `TWILIO_AUTH_TOKEN` (HMAC-SHA1 of the URL plus the sorted parameters). A request that fails the check gets a 403 and is not processed.
- **Not configured:** with no `TWILIO_*` set, the endpoint returns 503 and does nothing.
- **Port:** a leading port word picks the port (`MOSS`, `MONTEREY`, `SANTACRUZ`, `HMB`/`PRINCETON`). `HOME MOSS` saves a home port. Without either, the reply asks which port.
- **Reply size:** at most 2 GSM-7 segments (306 characters), ending with a source tag `[Src: <source> <M/D>]`. The tag adds `unverified` when the reply rests on official records nobody has verified yet.
- **Safety:** the same deterministic safety guard as the web chat applies. Questions about eating seafood, going out, crossing a bar or whether fishing is allowed get the fixed referral template.
- **Opt-out:** opted-out numbers get no reply to anything except HELP and START.
- **Rate limit:** 10 texts per hour per number (per server instance).
- **Durable store needed (production blocker):** opt-out and home port are kept **in memory** today, so they are lost on redeploy. That is acceptable for team phones on trial. Before any public number goes live, it needs a durable store (a Supabase table owned by the accounts track) behind `SmsStore` in `src/pilot/ask/smsHandler.ts`.
- **Logging:** if logging is configured, each question is logged with a salted hash of the phone number, never the number itself (see [ask.md](ask.md)).
