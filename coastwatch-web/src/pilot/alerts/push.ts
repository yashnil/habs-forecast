"use client";

import { SW_URL } from "@/v4/offline/useOfflinePack";

/**
 * Web push on this device. Gated three ways, each said honestly on the page:
 *  1. readiness("web_push") is live (VAPID keys configured on the server);
 *  2. the browser can do push (iOS/iPadOS only from a home-screen app, 16.4+);
 *  3. Neritic's service worker includes the push handler (public/sw-push.js, imported by sw.js).
 */
export type PushEnv = {
  supported: boolean;
  ios: boolean;
  /** opened from the home screen (installed web app) */
  standalone: boolean;
  permission: NotificationPermission | "unsupported";
};

export function pushEnv(): PushEnv {
  if (typeof window === "undefined") return { supported: false, ios: false, standalone: false, permission: "unsupported" };
  const ua = navigator.userAgent;
  const ios = /iPad|iPhone|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  let standalone = false;
  try {
    standalone = window.matchMedia("(display-mode: standalone)").matches || (navigator as unknown as { standalone?: boolean }).standalone === true;
  } catch {
    standalone = false;
  }
  const supported = "serviceWorker" in navigator && "PushManager" in window && "Notification" in window;
  return { supported, ios, standalone, permission: "Notification" in window ? Notification.permission : "unsupported" };
}

/** What to tell the person, in order of what stands in the way. */
export type PushAdvice = "not_configured" | "ios_install" | "unsupported" | "denied" | "ready" | "on";

export function pushAdvice(env: PushEnv, configured: boolean, subscribed: boolean): PushAdvice {
  if (!configured) return "not_configured";
  if (env.ios && !env.standalone) return "ios_install";
  if (!env.supported) return "unsupported";
  if (env.permission === "denied") return "denied";
  return subscribed && env.permission === "granted" ? "on" : "ready";
}

function keyBytes(base64url: string): Uint8Array<ArrayBuffer> {
  const pad = "=".repeat((4 - (base64url.length % 4)) % 4);
  const raw = atob((base64url + pad).replace(/-/g, "+").replace(/_/g, "/"));
  const out = new Uint8Array(new ArrayBuffer(raw.length));
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}

async function registration(): Promise<ServiceWorkerRegistration> {
  const existing = await navigator.serviceWorker.getRegistration("/");
  if (!existing) await navigator.serviceWorker.register(SW_URL, { scope: "/" });
  return navigator.serviceWorker.ready;
}

/** Does the active worker carry the push handler (sw-push.js)? Asks it; 1.5 s without an answer = no. */
export async function workerHandlesPush(): Promise<boolean> {
  try {
    const reg = await registration();
    const w = reg.active;
    if (!w) return false;
    return await new Promise<boolean>((resolve) => {
      const ch = new MessageChannel();
      const t = setTimeout(() => resolve(false), 1500);
      ch.port1.onmessage = (e) => {
        clearTimeout(t);
        resolve(!!(e.data as { push?: boolean } | null)?.push);
      };
      w.postMessage({ type: "neritic-push-ping" }, [ch.port2]);
    });
  } catch {
    return false;
  }
}

export async function currentSubscription(): Promise<PushSubscription | null> {
  try {
    const reg = await navigator.serviceWorker.getRegistration("/");
    return (await reg?.pushManager.getSubscription()) ?? null;
  } catch {
    return null;
  }
}

export async function subscribe(vapidPublicKey: string): Promise<PushSubscription> {
  const perm = await Notification.requestPermission();
  if (perm !== "granted") throw new Error(perm === "denied" ? "Notifications are blocked for this site in the browser settings." : "Notifications were not allowed.");
  const reg = await registration();
  return reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: keyBytes(vapidPublicKey) });
}

export async function unsubscribe(): Promise<void> {
  const s = await currentSubscription();
  await s?.unsubscribe();
}
