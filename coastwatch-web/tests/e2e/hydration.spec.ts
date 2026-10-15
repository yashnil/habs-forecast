import http from "node:http";
import type { AddressInfo } from "node:net";
import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * Hydration while the HTML is still arriving (React #418 regression). Pages are served through a
 * proxy that sends the HTML in small chunks with pauses, as a slow network does, while scripts
 * arrive at once; every cold load must hydrate without React's "server HTML didn't match" error.
 */
let proxy: http.Server;
let base = "";

test.beforeAll(async () => {
  const target = new URL(OK);
  proxy = http.createServer((req, res) => {
    const up = http.request({ host: target.hostname, port: target.port, path: req.url, method: req.method, headers: { ...req.headers, host: target.host, "accept-encoding": "identity" } }, async (r) => {
      if (!/text\/html/.test(String(r.headers["content-type"] ?? ""))) {
        res.writeHead(r.statusCode ?? 502, r.headers);
        r.pipe(res);
        return;
      }
      const headers = { ...r.headers };
      delete headers["content-length"];
      res.writeHead(r.statusCode ?? 502, headers);
      const parts: Buffer[] = [];
      for await (const c of r) parts.push(c as Buffer);
      const body = Buffer.concat(parts);
      for (let i = 0; i < body.length; i += 4096) {
        res.write(body.subarray(i, i + 4096));
        await new Promise((ok) => setTimeout(ok, 25));
      }
      res.end();
    });
    up.on("error", () => res.writeHead(502).end());
    req.pipe(up);
  });
  await new Promise<void>((ok) => proxy.listen(0, ok));
  base = `http://localhost:${(proxy.address() as AddressInfo).port}`;
});

test.afterAll(() => proxy.close());

test("pages hydrate cleanly when their HTML arrives slowly", async ({ browser }) => {
  test.setTimeout(240_000);
  const errors: string[] = [];
  for (const path of ["/", "/coast?place=port-550&for=commercial", "/fisheries", "/bloom", "/about"]) {
    for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
      for (let i = 0; i < 2; i++) {
        const ctx = await browser.newContext({ viewport });
        await ctx.addInitScript(() => {
          try {
            localStorage.setItem("cw-intro-dismissed-v1", "1");
          } catch {}
        });
        const page = await ctx.newPage();
        page.on("pageerror", (e) => /#418|#423|hydrat/i.test(String(e)) && errors.push(`${path} @${viewport.width}: ${String(e).slice(0, 90)}`));
        await page.goto(`${base}${path}`, { waitUntil: "networkidle" });
        await page.waitForTimeout(300);
        await ctx.close();
      }
    }
  }
  expect(errors).toEqual([]);
});
