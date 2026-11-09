"use client";

import { useEffect, useState } from "react";
import { usePilotAccount } from "@/pilot/account/context";
import { PUSH_CONSENT_TEXT } from "@/pilot/account/consent";
import { recordConsent, savePushEndpoint } from "@/pilot/account/sync";
import { currentSubscription, pushAdvice, pushEnv, subscribe, unsubscribe, workerHandlesPush, type PushAdvice, type PushEnv } from "./push";

/**
 * Notifications on this device, behind readiness("web_push"). Says exactly what stands in the way:
 * not connected yet, iPhone/iPad needs the home-screen app, unsupported browser, blocked, or ready.
 */
export function PushPanel() {
  const { config, session } = usePilotAccount();
  const [env, setEnv] = useState<PushEnv | null>(null);
  const [subscribed, setSubscribed] = useState(false);
  const [workerReady, setWorkerReady] = useState<boolean | null>(null);
  const [msg, setMsg] = useState("");
  const configured = !!config.vapidPublicKey;
  useEffect(() => {
    setEnv(pushEnv());
    if (configured) {
      void currentSubscription().then((s) => setSubscribed(!!s));
      void workerHandlesPush().then(setWorkerReady);
    }
  }, [configured]);
  if (!env) return null;
  const advice: PushAdvice = pushAdvice(env, configured, subscribed);
  const signedIn = session.status === "signed_in";

  const text: Record<PushAdvice, string> = {
    not_configured: "Phone and desktop notifications are not connected in this pilot build yet. Alerts appear in the inbox below (and on the bell at the top) when you open Neritic.",
    ios_install: "On iPhone and iPad, notifications work only from the home-screen app: tap Share, then “Add to Home Screen”, open Neritic from that icon, and come back here (iOS 16.4 or later).",
    unsupported: "This browser cannot show web notifications. The inbox still lists every alert.",
    denied: "Notifications are blocked for Neritic in this browser’s settings. Allow them there, then reload this page.",
    ready: "Get a notification on this device when an alert is added for a saved port.",
    on: "Notifications are on for this device.",
  };

  return (
    <div className="max-w-[68ch] space-y-3 text-[15px] leading-relaxed text-n-ink-2" data-testid="push-panel" data-advice={advice}>
      <p data-testid="push-advice">{text[advice]}</p>
      {advice === "ios_install" && (
        <ol className="list-decimal space-y-1 pl-5" data-testid="ios-steps">
          <li>Open this page in Safari.</li>
          <li>Tap the Share button, then “Add to Home Screen”.</li>
          <li>Open Neritic from the new icon and turn notifications on here.</li>
        </ol>
      )}
      {(advice === "ready" || advice === "on") && workerReady === false && <p className="text-n-ink">Notifications need an app update that is not in this build yet. The inbox still works.</p>}
      {advice === "ready" && workerReady !== false && (
        <>
          {!signedIn && <p>Sign in first: Neritic sends notifications to your account&rsquo;s devices.</p>}
          <p className="text-[13.5px]">{PUSH_CONSENT_TEXT}</p>
          <button
            type="button"
            disabled={!signedIn}
            className="today-btn primary nx-press nx-focus !min-h-12 px-5 disabled:opacity-50"
            onClick={async () => {
              try {
                const s = await subscribe(config.vapidPublicKey!);
                const r = await savePushEndpoint(session.client!, s.toJSON());
                if (!r.ok) throw new Error(r.error);
                await recordConsent(session.client!, "web_push", true, PUSH_CONSENT_TEXT);
                setSubscribed(true);
                setEnv(pushEnv());
                setMsg("Notifications are on. Choose “Notifications” under each port’s alerts.");
              } catch (e) {
                setEnv(pushEnv());
                setMsg((e as Error).message);
              }
            }}
            data-testid="push-enable"
          >
            Turn on notifications
          </button>
        </>
      )}
      {advice === "on" && (
        <button
          type="button"
          className="today-btn nx-press nx-focus nx-tint"
          onClick={async () => {
            await unsubscribe();
            if (session.client) await recordConsent(session.client, "web_push", false, PUSH_CONSENT_TEXT);
            setSubscribed(false);
            setMsg("Notifications are off for this device.");
          }}
        >
          Turn off
        </button>
      )}
      <p role="status">{msg}</p>
    </div>
  );
}
