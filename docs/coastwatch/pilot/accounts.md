# Pilot accounts, saved ports, alerts and legal pages (Track A)

Branch `pilot/accounts`. Everything works in the demo **without cloud credentials**. It becomes live by adding env vars; no rebuild is needed, because readiness is read per request.

## What is demo and what is live

| Feature | No env (demo default) | Local stack (`127.0.0.1`) | Cloud project |
|---|---|---|---|
| Readiness label | Accounts, notifications and texts show "Not yet connected" | "Pilot demo" | "Live" |
| Saved ports, fisheries, role, vessel class | Kept in this browser. The page says "Saved on this device — accounts not yet connected" | Email one-time-code sign-in; synced to the person's rows | same as local |
| In-app alerts | The device compares current PortToday facts with its last snapshot on each visit | Device check, plus the server fan-out after every refresh | same as local |
| Inbox | Device items | Device and account items, deduplicated | same as local |
| Web push | Explains why it is off | Needs VAPID keys **and** `sw-push.js` imported by `sw.js` | same as local |
| SMS | Consent is captured with its exact text. Nothing is sent | Consent is stored in `alert_subscriptions`. Nothing is sent | Sending belongs to Track Q, after 10DLC approval |
| Feedback, trip log | Kept on the device | Kept on the device and in the account | same as local |
| Legal pages | Live at `/privacy`, `/terms` and `/sms-terms` with the banner "Draft — pending owner/legal review" | | |

**Facts source.** Until `src/pilot/facts.ts` (pilot/today) is merged, alerts compare against **sample** facts in `src/pilot/alerts/factsStub.ts`:

- Every sample fact's source reads "Pilot sample (not live data)".
- The account page says so.
- The fan-out never sends web push from sample facts.

The swap is one function body in `src/pilot/alerts/source.ts`.

## Architecture

- `supabase/migrations/…_pilot_accounts.sql` creates these tables, each with RLS where users may only touch rows matching `user_id = (select auth.uid())`. `anon` is revoked.
  - `profiles` (the 13+ attestation is required)
  - `saved_places`
  - `alert_subscriptions` (`triggers[]`, `channel`; an SMS row needs `sms_phone_e164` plus a consent timestamp and its text)
  - `push_subscriptions`
  - `alerts` (the inbox; written by the server only; users may update only `read_at`)
  - `notifications_sent` (PK = subscription × fact id × fact version, the idempotency key)
  - `alert_fact_state` (server only)
  - `feedback`
  - `trip_logs`
  - `consent_log` (append-only)
  - `delete_my_account()`, which cascades to every row
- `…_pilot_retention.sql` adds `prune_pilot_data()`, which enforces the privacy policy's retention (inbox 180 days; feedback and trips 2 years after last sign-in).
- `supabase/tests/database/rls.test.sql`: 43 pgTAP tests (isolation between two users, anon denied, server-only tables, idempotency key, append-only consent, account deletion, retention).
- `src/pilot/alerts/diff.ts` holds the rules both sides share:
  - triggers map from fact kinds;
  - a fact version is a hash of its significant fields (a re-check date alone is not a change);
  - a first look only sets a baseline;
  - bloom risk counts only when it rises by ≥ 20 points or to amber/red;
  - a `bloom_outlook` (chlorophyll) never alerts;
  - a removed official record is "no longer listed; reason unknown".
- `src/pilot/alerts/fanout.ts` + `server.ts` + `POST /api/alerts/fanout` run the fan-out:
  1. claim in `notifications_sent`;
  2. write the alert;
  3. send push if VAPID is set (dead endpoints are deleted);
  4. advance the state.

  A rerun, or a crash between steps, cannot produce a duplicate. Protected by `Authorization: Bearer $ALERTS_FANOUT_SECRET`. The service-role key is used only here. Limitation: if a fact returns to a version it already had (A→B→A), the second A is not re-sent.
- `src/pilot/account/*`:
  - device storage with a try/catch fallback to memory (the page warns when storage is blocked);
  - the account model;
  - `@supabase/ssr` browser client;
  - `sync.ts`, which reads and writes the person's rows through RLS;
  - the `/account` UI;
  - `Feedback`, `TripLogForm`, `ReadinessBadge`;
  - `config.ts`, which computes readiness and public config on the server; secrets never reach the browser (unit-tested).

## Run the local stack (optional, proves live mode)

```sh
# repo root; ports 563xx, project_id neritic-pilot (never touches other local stacks)
npx supabase@2.121.0 start -x realtime,storage-api,imgproxy,studio,edge-runtime,logflare,vector,supavisor,postgres-meta
npx supabase@2.121.0 test db                     # pgTAP: 43/43
cd coastwatch-web && npm run build
NEXT_PUBLIC_SUPABASE_URL=http://127.0.0.1:56321 NEXT_PUBLIC_SUPABASE_ANON_KEY=<ANON_KEY from `supabase status`> \
SUPABASE_SERVICE_ROLE_KEY=<SERVICE_ROLE_KEY> ALERTS_FANOUT_SECRET=local-test-secret CW_DATA_BASE_URL=/data/fixture/v1 \
npx next start -p 3312
# sign-in codes arrive in Mailpit: http://127.0.0.1:56324
PILOT_LIVE_BASE=http://localhost:3312 SUPABASE_SERVICE_ROLE_KEY=… ALERTS_FANOUT_SECRET=local-test-secret \
CW_E2E_PORT_BASE=3330 npx playwright test tests/e2e/pilot-account-live.spec.ts
npx supabase@2.121.0 stop --project-id neritic-pilot
```

## Going live (owner)

1. **Create the Supabase project** in **us-west-1 (N. California)**. Use **Pro ($25/month)**: the free tier pauses after a week idle, which is fatal for a pilot. This is spend and needs your approval.
2. **Auth → Email:**
   - turn email OTP on;
   - set the "Magic Link" and "Confirm signup" templates to `supabase/templates/otp.html` (it shows `{{ .Token }}`);
   - OTP expiry 600 s;
   - site URL = the preview URL.
   - Add custom SMTP (for example Resend or Postmark) before real users. The built-in sender is rate-limited to a few emails an hour.
3. **Run the migrations:** `npx supabase@2.121.0 link --project-ref <ref>`, then `npx supabase@2.121.0 db push`.
4. **Set env** on the private preview (see `coastwatch-web/.env.example`):
   - `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - `SUPABASE_SERVICE_ROLE_KEY` (server only)
   - `ALERTS_FANOUT_SECRET`
   - optionally `NEXT_PUBLIC_VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY`, `VAPID_SUBJECT` (`npx web-push@3.6.7 generate-vapid-keys`)
   - `CW_SITE_URL`
5. **Enable pg_cron and pg_net** (Database → Extensions), store the URL and secret in Vault, and run the commented `cron.schedule(...)` blocks at the end of both migrations: the fan-out hourly at :20, retention daily.
6. **Legal pages:**
   - replace `[OWNER: add a contact email before launch]` in `src/pilot/account/LegalPage.tsx`;
   - have a person (ideally counsel) read `/privacy`, `/terms` and `/sms-terms`;
   - then remove the draft banner and bump `LEGAL_VERSION` in `src/pilot/account/consent.ts`.
7. SMS stays off until the Twilio 10DLC campaign is approved (Track Q).

## Integrator notes (shared files)

- **TopBar** (`src/v4/shell/TopBar.tsx`): add one line next to `<MoreMenu />`: `<AlertsBell />`, from `@/pilot/alerts/AlertsBell`. It works with no provider (device inbox only).
- **Today page:**
  - mount `<AlertCheck todays={[today]} />` (from `@/pilot/alerts/AlertCheck`) to run the device alert check;
  - mount `<Feedback factId={fact.id} placeId={fact.place_id} />` under fact cards;
  - optionally wrap the page in `<PilotAccountProvider config={pilotAccountConfig()}>` so feedback syncs when the person is signed in.
- **`public/sw.js`:** add `importScripts("/sw-push.js");` after the constants. It adds only push, notificationclick and a ping listener. Until then, the account page says notifications are not in this build.
- **`package.json`:**
  - dependencies: `@supabase/supabase-js@2.118.0`, `@supabase/ssr@0.12.7`, `web-push@3.6.7`
  - devDependencies: `@types/web-push@3.6.4`
- **Facts:** replace the body of `loadPortTodays()` in `src/pilot/alerts/source.ts` with the real `facts.ts` builder and return `stub: false`.
- A link to `/account` (for example "Your ports and alerts" in the ⋯ menu) is the integrator's call.
