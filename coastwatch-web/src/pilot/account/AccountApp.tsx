"use client";

import Link from "next/link";
import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { PILOT_PORTS, type PortToday } from "@/pilot/contracts";
import { TRIGGERS, TRIGGER_LABELS } from "@/pilot/alerts/diff";
import { clearInbox, runLocalCheck } from "@/pilot/alerts/inbox";
import { InboxList } from "@/pilot/alerts/InboxList";
import { useInbox } from "@/pilot/alerts/useInbox";
import { PushPanel } from "@/pilot/alerts/PushPanel";
import type { PilotAccountConfig } from "./config";
import { PilotAccountProvider, usePilotAccount } from "./context";
import { PilotHero } from "@/pilot/shell/PilotShell";
import { AGE_TEXT, SMS_CONSENT_TEXT, toE164US } from "./consent";
import { useAccount } from "./hooks";
import { FISHERIES, ROLES, VESSEL_CLASSES, forgetDevice, placeLabel, readAccount, removePlace, savePlace, setChannel, setTrigger, updateAccount, type LocalAccount } from "./local";
import { ReadinessBadge } from "./ReadinessBadge";
import { deleteAccount, pullAccount, pushAccount, recordConsent } from "./sync";
import { TripLogForm } from "./TripLogForm";

/**
 * /account: sign-in (email one-time code) when accounts are configured; otherwise the same choices
 * kept on this device, labelled so. Saved ports, fisheries, alert preferences, the alert inbox,
 * notifications, an optional trip log, and the person's data (delete on device / delete account).
 */
export function AccountApp({ config, todays, sample }: { config: PilotAccountConfig; todays: PortToday[]; /** the facts are the test fixture's, not published data */ sample: boolean }) {
  return (
    <PilotAccountProvider config={config}>
      <Account todays={todays} sample={sample} />
    </PilotAccountProvider>
  );
}

// the Almanac controls (src/pilot/shell/pilot.css, src/pilot/today/today.css): square shoulders, 44 px or more
const field = "pilot-field nx-focus";
const primary = "today-btn primary nx-press nx-focus !min-h-12 px-5 disabled:opacity-50";
const quiet = "today-btn nx-press nx-focus nx-tint";
const check = "pilot-check";
const link = "pilot-link nx-focus";
const label = "pilot-label";

function Section({ id, title, badge, children, lede }: { id: string; title: string; badge?: ReactNode; lede?: ReactNode; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="today-sec" data-testid={`account-${id}`}>
      <header>
        <h2 id={`${id}-h`} className="today-h2">
          {title}
        </h2>
        {badge}
      </header>
      {lede && <p className="pilot-sec-lede">{lede}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

function Account({ todays, sample }: { todays: PortToday[]; sample: boolean }) {
  const { config, session } = usePilotAccount();
  const { account, hydrated, persistent } = useAccount();
  const inbox = useInbox(session);
  const [syncNote, setSyncNote] = useState<string | null>(null);
  const configured = !!config.supabase;
  const signedIn = session.status === "signed_in";

  // the device's alert check for saved ports (idempotent)
  const saved = account.saved_places.join(",");
  const prefs = JSON.stringify(account.alerts);
  useEffect(() => {
    if (hydrated) runLocalCheck(account, todays, new Date(), { sample });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, saved, prefs]);

  // signed in: keep the account's rows in step with the device copy (debounced)
  const first = useRef(true);
  useEffect(() => {
    if (!signedIn || !session.client || !hydrated || !account.age_13_plus) return;
    if (first.current) {
      first.current = false;
      return;
    }
    const t = setTimeout(() => {
      void pushAccount(session.client!, account).then((r) => setSyncNote(r.ok ? "Saved to your account." : `Kept on this device; not synced (${r.error}).`));
    }, 600);
    return () => clearTimeout(t);
  }, [account, signedIn, session.client, hydrated]);

  // "Set an alert" on Today arrives as /account?place=port-592: that port is put first, one tap from its alerts
  const [want, setWant] = useState<string | null>(null);
  useEffect(() => {
    const p = new URLSearchParams(window.location.search).get("place");
    if (p && PILOT_PORTS.some((x) => x.place_id === p)) setWant(p);
  }, []);
  const wantSaved = !!want && account.saved_places.includes(want);
  const toAlerts = (place: string) => requestAnimationFrame(() => document.getElementById(`alerts-${place}`)?.scrollIntoView({ block: "start" }));

  const mode = !configured ? "device" : signedIn ? "account" : "signed_out";
  const edit = (fn: (a: LocalAccount) => LocalAccount) => updateAccount(fn);

  return (
    <div data-testid="account-page" data-mode={mode}>
      <PilotHero eyebrow="Neritic private pilot" title="Your ports and alerts" lede="The ports you follow, what should alert you, and the alerts themselves. Official status always comes first.">
        <ul className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2" aria-label="What is connected">
          {(
            [
              ["Accounts", config.readiness.accounts, "readiness-accounts"],
              ["Notifications", config.readiness.web_push, "readiness-push"],
              ["Text alerts", config.readiness.sms, "readiness-sms"],
            ] as const
          ).map(([name, value, testId]) => (
            <li key={name} className="flex items-center gap-2 whitespace-nowrap text-[14px] text-n-ink-2">
              {name} <ReadinessBadge value={value} testId={testId} />
            </li>
          ))}
        </ul>
        <div className="mt-4 space-y-3">
          {want && hydrated && (
            <div className="pilot-note" data-testid="want-port" data-place={want} data-saved={wantSaved ? "1" : "0"}>
              {wantSaved ? (
                <p>
                  <strong className="font-semibold">{placeLabel(want)} is saved on this device.</strong>{" "}
                  <a href={`#alerts-${want}`} className={link} data-testid="want-alerts-link">
                    Choose what alerts you
                  </a>{" "}
                  below.
                </p>
              ) : (
                <>
                  <p>
                    <strong className="font-semibold">Alerts for {placeLabel(want)}.</strong> Save the port, then choose what should alert you: official status changes, new toxin tests, whale-rule actions.
                  </p>
                  <button
                    type="button"
                    className={`${primary} mt-3`}
                    onClick={() => {
                      edit((a) => savePlace(a, want));
                      toAlerts(want);
                    }}
                    data-testid="want-save"
                  >
                    Save {placeLabel(want)} and set alerts
                  </button>
                </>
              )}
            </div>
          )}
          {!configured && (
            <p className="pilot-note" data-testid="device-only-banner">
              <strong className="font-semibold">Saved on this device — accounts not yet connected.</strong> Your ports, fisheries, alert choices and inbox stay in this browser. Nothing is sent anywhere. Clearing the browser&rsquo;s data removes them.
            </p>
          )}
          {hydrated && !persistent && (
            <p role="alert" className="pilot-note amber" data-testid="storage-blocked">
              This browser is not letting Neritic keep anything (private window or blocked site data). Your choices work for this visit only.
            </p>
          )}
          {syncNote && signedIn && (
            <p role="status" className="today-src" data-testid="sync-note">
              {syncNote}
            </p>
          )}
        </div>
      </PilotHero>
      <div className="today-wrap">
      {/* one column on a phone; on a wide screen the inbox sits beside the settings, as Today's columns do (pilot.css) */}
      <div className="account-grid">

      <Section id="signin" title={configured ? (signedIn ? "Signed in" : "Sign in") : "Set up on this device"} lede={configured ? (signedIn ? null : "We email you a 6-digit code. No password.") : "Confirm your age once; everything else is optional."}>
        <SignIn account={account} hydrated={hydrated} />
      </Section>

      <Section id="profile" title="Your fishing" lede="Used to put your fisheries first. It changes no value, caveat or official status.">
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="acct-role" className={label}>You are</label>
            <select id="acct-role" value={account.role ?? ""} onChange={(e) => edit((a) => ({ ...a, role: (e.target.value || null) as LocalAccount["role"] }))} className={field} data-testid="acct-role">
              <option value="">Choose…</option>
              {ROLES.map((r) => (
                <option key={r.id} value={r.id}>{r.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="acct-vessel" className={label}>Vessel length (optional)</label>
            <select id="acct-vessel" value={account.vessel_length_class ?? ""} onChange={(e) => edit((a) => ({ ...a, vessel_length_class: (e.target.value || null) as LocalAccount["vessel_length_class"] }))} className={field}>
              <option value="">—</option>
              {VESSEL_CLASSES.map((v) => (
                <option key={v.id} value={v.id}>{v.label}</option>
              ))}
            </select>
          </div>
        </div>
        <fieldset className="mt-5">
          <legend className={label}>Fisheries you work</legend>
          <div className="grid gap-x-6 gap-y-1 sm:grid-cols-2" data-testid="acct-fisheries">
            {FISHERIES.map((f) => (
              <label key={f.id} className="flex min-h-11 cursor-pointer items-start gap-3 py-1.5 text-[15px] text-n-ink">
                <input
                  type="checkbox"
                  className={check}
                  checked={account.fisheries.includes(f.id)}
                  onChange={(e) => edit((a) => ({ ...a, fisheries: e.target.checked ? [...new Set([...a.fisheries, f.id])] : a.fisheries.filter((x) => x !== f.id) }))}
                />
                {f.label}
              </label>
            ))}
          </div>
        </fieldset>
      </Section>

      <Section id="ports" title="Saved ports" lede="Pilot ports first. Your home port opens first on Today.">
        <ul data-testid="acct-ports">
          {PILOT_PORTS.map((p) => {
            const on = account.saved_places.includes(p.place_id);
            const home = account.home_place_id === p.place_id;
            return (
              <li key={p.place_id} className="flex flex-wrap items-center justify-between gap-2 border-b border-n-line py-3" data-testid={`port-${p.place_id}`} data-saved={on ? "1" : "0"} data-want={want === p.place_id ? "1" : undefined}>
                <div className="min-w-0">
                  <span className="font-serif text-[22px] leading-tight text-n-ink">{p.label}</span>
                  {home && <span className="nx-caps ml-2">Home port</span>}
                  <Link href={`/today?place=${p.place_id}`} className={`${link} ml-3 inline-flex min-h-11 items-center text-[14px]`}>
                    Today<span className="sr-only"> at {p.label}</span>
                  </Link>
                </div>
                <div className="flex gap-2">
                  {on && !home && (
                    <button type="button" className={quiet} onClick={() => edit((a) => ({ ...a, home_place_id: p.place_id }))}>
                      Make home<span className="sr-only"> port: {p.label}</span>
                    </button>
                  )}
                  <button type="button" aria-pressed={on} className={on ? quiet : primary} onClick={() => edit((a) => (on ? removePlace(a, p.place_id) : savePlace(a, p.place_id)))} data-testid={`save-${p.place_id}`}>
                    {on ? "Saved" : "Save"}
                    <span className="sr-only"> {p.label}</span>
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      </Section>

      <Section
        id="alerts"
        title="Alerts"
        badge={<ReadinessBadge value={config.readiness.alerts === "not_configured" ? "demo" : config.readiness.alerts} testId="readiness-alerts" />}
        lede={
          configured
            ? "In-app alerts are checked on this device each visit and, when you are signed in, by Neritic after every data refresh."
            : "In-app alerts are checked on this device each time you open Neritic. Without an account nothing is checked while Neritic is closed."
        }
      >
        {sample && (
          <p className="pilot-note model mb-4" data-testid="sample-facts-note">
            Sample data: this build is reading Neritic&rsquo;s test fixture, not today&rsquo;s published data. Alerts made from it are marked &ldquo;Sample data&rdquo; and are never sent to a phone.
          </p>
        )}
        {!account.saved_places.length ? (
          <p className="text-[14.5px] text-n-ink-2">
            Save a port above to choose its alerts. Every current official notice on the coast is listed on{" "}
            <a href="/notices" className={link} data-testid="notices-all">
              one page
            </a>
            .
          </p>
        ) : (
          <div className="space-y-6">
            {account.saved_places.map((place) => {
              const pref = account.alerts.find((p) => p.place_id === place);
              return (
                <fieldset key={place} id={`alerts-${place}`} className="scroll-mt-20 border-t border-n-line-strong pt-3" data-testid={`alerts-${place}`}>
                  <legend className="float-left mb-2 w-full font-serif text-[22px] leading-tight text-n-ink">{placeLabel(place)}</legend>
                  <p className="clear-both mb-2 text-[14.5px]">
                    <a href={`/notices?place=${encodeURIComponent(place)}`} className={link} data-testid={`notices-${place}`}>
                      Every official notice at {placeLabel(place)} and on the rest of the coast
                    </a>
                  </p>
                  <div className="clear-both grid gap-x-6 sm:grid-cols-2">
                    {TRIGGERS.map((t) => (
                      <label key={t} className="flex min-h-11 cursor-pointer items-start gap-3 py-1.5">
                        <input type="checkbox" className={check} checked={!!pref?.triggers.includes(t)} onChange={(e) => edit((a) => setTrigger(a, place, t, e.target.checked))} data-testid={`trigger-${place}-${t}`} />
                        <span>
                          <span className="block text-[15px] text-n-ink">{TRIGGER_LABELS[t].label}</span>
                          <span className="block text-[13px] leading-snug text-n-ink-2">{TRIGGER_LABELS[t].hint}</span>
                        </span>
                      </label>
                    ))}
                  </div>
                  <div className="mt-3 border-t border-n-line pt-3">
                    <p className={label}>How</p>
                    <div className="flex flex-wrap gap-x-6 gap-y-1">
                      <label className="flex min-h-11 items-center gap-3 text-[15px] text-n-ink">
                        <input type="checkbox" className={check} checked={!!pref?.channels.includes("in_app")} onChange={(e) => edit((a) => setChannel(a, place, "in_app", e.target.checked))} />
                        In the app (inbox)
                      </label>
                      <label className="flex min-h-11 items-center gap-3 text-[15px] text-n-ink">
                        <input type="checkbox" className={check} disabled={!config.vapidPublicKey || !signedIn} checked={!!pref?.channels.includes("web_push")} onChange={(e) => edit((a) => setChannel(a, place, "web_push", e.target.checked))} />
                        Notifications
                        {(!config.vapidPublicKey || !signedIn) && <span className="text-[13px] text-n-ink-2">({config.vapidPublicKey ? "sign in first" : "not yet connected"})</span>}
                      </label>
                      <label className="flex min-h-11 items-center gap-3 text-[15px] text-n-ink">
                        <input type="checkbox" className={check} disabled={config.readiness.sms !== "live" || !account.sms || !signedIn} checked={!!pref?.channels.includes("sms")} onChange={(e) => edit((a) => setChannel(a, place, "sms", e.target.checked))} />
                        Text message
                        {config.readiness.sms !== "live" && <span className="text-[13px] text-n-ink-2">(not yet available: carrier registration pending)</span>}
                      </label>
                    </div>
                  </div>
                </fieldset>
              );
            })}
          </div>
        )}
      </Section>

      <Section id="push" title="Notifications on this device" badge={<ReadinessBadge value={config.readiness.web_push} />}>
        <PushPanel />
      </Section>

      <Section
        id="inbox"
        title="Inbox"
        badge={inbox.unread ? <span className="text-[13.5px] font-semibold text-n-ink" data-testid="inbox-unread">{inbox.unread} unread</span> : null}
        lede="Each alert says what changed, its source and the date that applies. Official status comes first. Nothing here says waters are safe."
      >
        <InboxList items={inbox.items} />
        {inbox.items.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {inbox.unread > 0 && (
              <button type="button" className={quiet} onClick={inbox.markAllRead} data-testid="inbox-mark-read">
                Mark all read
              </button>
            )}
            <button type="button" className={quiet} onClick={() => clearInbox()} data-testid="inbox-clear">
              Clear alerts on this device
            </button>
          </div>
        )}
      </Section>

      <Section id="trips" title="Trip feedback (pilot)" lede="A few taps after a trip tells the pilot team whether Neritic was useful. It is sent only when you are signed in. No catch numbers or fishing spots.">
        <TripLogForm defaultPlace={account.home_place_id} />
      </Section>

      <Section id="data" title="Your data">
        <DataControls />
      </Section>

      </div>
      </div>
    </div>
  );
}

function SignIn({ account, hydrated }: { account: LocalAccount; hydrated: boolean }) {
  const { config, session } = usePilotAccount();
  const id = useId();
  const [age, setAge] = useState(false);
  const [smsOn, setSmsOn] = useState(false);
  const [phone, setPhone] = useState("");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [step, setStep] = useState<"email" | "code">("email");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const configured = !!config.supabase;
  const attested = !!account.age_13_plus;

  /** the consents on the form, applied to the device account (and logged to the account when signed in) */
  function applyConsents(): string | null {
    if (!attested && !age) return "Please confirm you are 13 or older.";
    let sms: LocalAccount["sms"] = account.sms;
    if (smsOn) {
      const e164 = toE164US(phone);
      if (!e164) return "Enter a 10-digit US mobile number for text alerts, or untick the text-alert box.";
      sms = { phone_e164: e164, consent: { at: new Date().toISOString(), text: SMS_CONSENT_TEXT } };
    }
    const now = new Date().toISOString();
    updateAccount((a) => ({ ...a, age_13_plus: a.age_13_plus ?? { at: now, text: AGE_TEXT }, sms, created_at: a.created_at ?? now }));
    return null;
  }

  async function logConsents() {
    if (!session.client) return;
    if (age) await recordConsent(session.client, "age_13_plus", true, AGE_TEXT);
    if (smsOn) await recordConsent(session.client, "sms", true, SMS_CONSENT_TEXT);
  }

  const consentFields = (
    <div className="space-y-3">
      {!attested && (
        <label className="flex cursor-pointer items-start gap-3 text-[15px] leading-snug text-n-ink">
          <input type="checkbox" required className={check} checked={age} onChange={(e) => setAge(e.target.checked)} data-testid="consent-age" />
          <span>
            I am 13 or older, and I have read the{" "}
            <Link href="/terms" className={link}>Terms of Use</Link> and the{" "}
            <Link href="/privacy" className={link}>Privacy Policy</Link>.
          </span>
        </label>
      )}
      <div className="border-l-2 border-n-line-strong pl-3">
        <label className="flex cursor-pointer items-start gap-3 text-[14.5px] leading-snug text-n-ink">
          <input type="checkbox" className={check} checked={smsOn} onChange={(e) => setSmsOn(e.target.checked)} data-testid="consent-sms" aria-describedby={`${id}-sms-note`} />
          <span data-testid="sms-disclosure">{SMS_CONSENT_TEXT}</span>
        </label>
        <p id={`${id}-sms-note`} className="mt-1 pl-8 text-[13px] text-n-ink-2">
          Optional. <Link href="/sms-terms" className={link}>SMS Terms</Link>. Text alerts are not sent yet (carrier registration pending); your choice is recorded for when they are.
        </p>
        {smsOn && (
          <div className="mt-2 pl-8">
            <label htmlFor={`${id}-phone`} className={label}>Mobile number (US)</label>
            <input id={`${id}-phone`} type="tel" inputMode="tel" autoComplete="tel-national" value={phone} onChange={(e) => setPhone(e.target.value)} className={`${field} max-w-[240px]`} data-testid="sms-phone" />
          </div>
        )}
      </div>
    </div>
  );

  if (!hydrated) return <p className="text-[14.5px] text-n-ink-2">Loading…</p>;

  // ------------------------------------------------------------- accounts not configured
  // Nothing to set up: with no account there is no age confirmation and no phone number to give. The page is
  // only served where accounts are connected (src/app/account/pilot/page.tsx); this is the safe answer if not.
  if (!configured) {
    return (
      <p className="text-[15px] text-n-ink" data-testid="device-setup">
        Accounts are not connected here, so there is nothing to sign in to. Your ports, alerts and preferences are kept in this browser:{" "}
        <a href="/account" className={link}>
          Your data and preferences
        </a>
        .
      </p>
    );
  }

  // ------------------------------------------------------------- signed in
  if (session.status === "signed_in") {
    return (
      <div className="space-y-3" data-testid="signed-in">
        <p className="text-[15px] text-n-ink">
          Signed in as <strong className="font-semibold">{session.user?.email}</strong>.
        </p>
        <button type="button" className={quiet} onClick={() => session.client?.auth.signOut()}>
          Sign out
        </button>
        <Msg msg={msg} />
      </div>
    );
  }

  // ------------------------------------------------------------- sign in with an email code
  return (
    <div data-testid="signin-form">
      {step === "email" ? (
        <form
          className="space-y-4"
          onSubmit={async (e) => {
            e.preventDefault();
            const err = applyConsents();
            if (err) return setMsg({ tone: "error", text: err });
            setBusy(true);
            const r = await session.client!.auth.signInWithOtp({ email: email.trim(), options: { shouldCreateUser: true } });
            setBusy(false);
            if (r.error) return setMsg({ tone: "error", text: `Could not send the code: ${r.error.message}` });
            setStep("code");
            setMsg({ tone: "ok", text: `We sent a 6-digit code to ${email.trim()}. It expires in 10 minutes.` });
          }}
        >
          <div>
            <label htmlFor={`${id}-email`} className={label}>Email</label>
            <input id={`${id}-email`} type="email" required autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} className={`${field} max-w-[360px]`} data-testid="signin-email" />
          </div>
          {consentFields}
          <button type="submit" disabled={busy} className={primary} data-testid="signin-send">
            {busy ? "Sending…" : "Email me a code"}
          </button>
        </form>
      ) : (
        <form
          className="space-y-4"
          onSubmit={async (e) => {
            e.preventDefault();
            setBusy(true);
            const r = await session.client!.auth.verifyOtp({ email: email.trim(), token: code.trim(), type: "email" });
            if (r.error) {
              setBusy(false);
              return setMsg({ tone: "error", text: `That code did not work: ${r.error.message}` });
            }
            // a returning person on a new device: bring their ports and alerts here, keep anything added on this device
            const remote = await pullAccount(session.client!);
            if (remote) {
              updateAccount((a) => ({
                ...a,
                role: a.role ?? remote.role ?? null,
                vessel_length_class: a.vessel_length_class ?? remote.vessel_length_class ?? null,
                home_place_id: a.home_place_id ?? remote.home_place_id ?? null,
                age_13_plus: a.age_13_plus ?? remote.age_13_plus ?? null,
                sms: a.sms ?? remote.sms ?? null,
                fisheries: a.fisheries.length ? a.fisheries : (remote.fisheries ?? []),
                saved_places: [...new Set([...a.saved_places, ...(remote.saved_places ?? [])])],
                alerts: [...a.alerts, ...(remote.alerts ?? []).filter((r) => !a.alerts.some((x) => x.place_id === r.place_id))],
              }));
            }
            await logConsents();
            const pushed = await pushAccount(session.client!, readAccount());
            setBusy(false);
            setMsg(pushed.ok ? { tone: "ok", text: "Signed in. Your ports and alerts are saved to your account." } : { tone: "error", text: `Signed in, but saving to your account failed: ${pushed.error}` });
          }}
        >
          <div>
            <label htmlFor={`${id}-code`} className={label}>6-digit code</label>
            <input id={`${id}-code`} required inputMode="numeric" autoComplete="one-time-code" pattern="[0-9]{6}" maxLength={6} value={code} onChange={(e) => setCode(e.target.value)} className={`${field} max-w-[180px] tracking-[0.3em]`} data-testid="signin-code" />
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="submit" disabled={busy} className={primary} data-testid="signin-verify">
              {busy ? "Checking…" : "Sign in"}
            </button>
            <button type="button" className={quiet} onClick={() => setStep("email")}>
              Use a different email
            </button>
          </div>
        </form>
      )}
      <Msg msg={msg} />
    </div>
  );
}

function Msg({ msg }: { msg: { tone: "ok" | "error"; text: string } | null }) {
  return (
    <p role={msg?.tone === "error" ? "alert" : "status"} className={`mt-3 text-[14.5px] ${msg?.tone === "error" ? "font-medium t-red" : "text-n-ink-2"}`} data-testid="signin-msg">
      {msg?.text ?? ""}
    </p>
  );
}

function DataControls() {
  const { session } = usePilotAccount();
  const [confirm, setConfirm] = useState<"device" | "account" | null>(null);
  const [msg, setMsg] = useState("");
  const signedIn = session.status === "signed_in";
  return (
    <div className="space-y-4 text-[14.5px] leading-relaxed text-n-ink-2">
      <p>
        {signedIn
          ? "Your profile, saved ports, alert choices, inbox, feedback and trip logs are stored in your Neritic account (Supabase, US West) and on this device. Nothing is sold or shared for marketing."
          : "Everything on this page is kept in this browser only. Neritic has no copy."}{" "}
        See the <Link href="/privacy" className={link}>Privacy Policy</Link> for what is kept and for how long.
      </p>
      <div className="flex flex-wrap gap-2">
        <button type="button" className={quiet} onClick={() => setConfirm("device")} data-testid="forget-device">
          Delete everything on this device
        </button>
        {signedIn && (
          <button type="button" className={`${quiet} !border-[var(--t-red)] t-red`} onClick={() => setConfirm("account")} data-testid="delete-account">
            Delete my account
          </button>
        )}
      </div>
      {confirm && (
        <div role="alertdialog" aria-labelledby="confirm-h" className="pilot-note red" data-testid="confirm-delete">
          <p id="confirm-h" className="font-semibold text-n-ink">
            {confirm === "device" ? "Delete your ports, alerts, inbox, feedback and trips from this device?" : "Delete your Neritic account and everything in it? This cannot be undone."}
          </p>
          <div className="mt-3 flex gap-2">
            <button
              type="button"
              className={`${primary} !border-[var(--t-red)] !bg-[var(--t-red)]`}
              onClick={async () => {
                if (confirm === "account" && session.client) {
                  const r = await deleteAccount(session.client);
                  if (!r.ok) {
                    setMsg(`Could not delete the account: ${r.error}`);
                    setConfirm(null);
                    return;
                  }
                }
                forgetDevice();
                setMsg(confirm === "account" ? "Your account and this device's copy are deleted." : "Deleted from this device.");
                setConfirm(null);
              }}
              data-testid="confirm-delete-yes"
            >
              Delete
            </button>
            <button type="button" className={quiet} onClick={() => setConfirm(null)}>
              Cancel
            </button>
          </div>
        </div>
      )}
      <p role="status" data-testid="data-msg">{msg}</p>
    </div>
  );
}
