/*
 * Neritic pilot: web push handler for the service worker.
 *
 * INTEGRATOR: add ONE line near the top of public/sw.js (after the constants):
 *     importScripts("/sw-push.js");
 * It only adds "push", "notificationclick" and a "neritic-push-ping" message listener; it never
 * touches fetch handling or caches. Until it is imported, the account page says notifications are
 * not available on this build (src/pilot/alerts/push.ts → workerHandlesPush).
 *
 * Payload (from /api/alerts/fanout): { title, body, url, tag }. A notification only repeats what the
 * inbox says; it never says "safe".
 */
self.addEventListener("push", (event) => {
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch {
    data = { body: event.data ? event.data.text() : "" };
  }
  const title = typeof data.title === "string" && data.title ? data.title.slice(0, 120) : "Neritic alert";
  const body = typeof data.body === "string" ? data.body.slice(0, 240) : "Something changed at a port you follow.";
  const url = typeof data.url === "string" && data.url.startsWith(self.location.origin) ? data.url : "/account#inbox";
  event.waitUntil(
    self.registration.showNotification(title, {
      body,
      tag: typeof data.tag === "string" ? data.tag : undefined,
      icon: "/icons/icon-192.png",
      badge: "/icons/icon-192.png",
      data: { url },
    }),
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = (event.notification.data && event.notification.data.url) || "/account#inbox";
  event.waitUntil(
    (async () => {
      const all = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
      for (const c of all) {
        if (new URL(c.url).origin === self.location.origin && "focus" in c) {
          await c.focus();
          if ("navigate" in c) await c.navigate(url);
          return;
        }
      }
      await self.clients.openWindow(url);
    })(),
  );
});

self.addEventListener("message", (e) => {
  if (e.data && e.data.type === "neritic-push-ping" && e.ports && e.ports[0]) e.ports[0].postMessage({ push: true });
});
