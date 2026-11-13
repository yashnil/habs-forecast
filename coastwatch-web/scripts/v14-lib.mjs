/**
 * Shared pieces of the v14 visual comparison (scripts/v14-reference-shots.mjs, scripts/v14-app-shots.mjs):
 * the five frames, the five sizes, a static server for the prototype folder, and WebP encoding done by the
 * browser itself (no image library needed).
 */
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";

export const FRAMES = ["trip-empty", "trip-done", "today", "research", "alerts"];

/** The owner's five review sizes (desktop, laptop, small laptop, phone, small phone). Device scale 1. */
export const SIZES = [
  [1440, 900],
  [1280, 720],
  [1100, 800],
  [390, 844],
  [320, 640],
];
export const PHONE_MAX = 640;
export const sizeName = ([w, h]) => `${w}x${h}`;
export const contextFor = ([w, h], extra = {}) => ({
  viewport: { width: w, height: h },
  deviceScaleFactor: 1,
  ...(w <= PHONE_MAX ? { isMobile: true, hasTouch: true } : {}),
  ...extra,
});

/** Filter by env: FRAMES=today,research SIZES=390x844,320x640 */
export const pickFrames = () => (process.env.FRAMES ? FRAMES.filter((f) => process.env.FRAMES.split(",").includes(f)) : FRAMES);
export const pickSizes = () => (process.env.SIZES ? SIZES.filter((s) => process.env.SIZES.split(",").includes(sizeName(s))) : SIZES);

export const PROTOTYPE_DIR =
  process.env.CW_V14_PROTOTYPE ??
  "/private/tmp/claude-502/-Users-yashnilmohanty-Desktop-habs-forecast/597c2abb-016b-4c09-bb3a-b1e2d415299e/scratchpad/art/concept-4-synthesis";

const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".png": "image/png", ".json": "application/json", ".svg": "image/svg+xml", ".woff2": "font/woff2", ".webm": "video/webm" };

/** Serve a folder on 127.0.0.1:<port>. Resolves to the server (call .close()). */
export function serveFolder(dir, port) {
  const root = resolve(dir);
  const server = createServer(async (req, res) => {
    let p = decodeURIComponent((req.url ?? "/").split("?")[0]);
    if (p.endsWith("/")) p += "index.html";
    const file = join(root, normalize(p));
    if (!file.startsWith(root)) return res.writeHead(403).end();
    try {
      const buf = await readFile(file);
      res.writeHead(200, { "Content-Type": TYPES[extname(file)] ?? "application/octet-stream", "Cache-Control": "no-store" }).end(buf);
    } catch {
      res.writeHead(404).end("not found");
    }
  });
  return new Promise((ok, bad) => {
    server.once("error", bad);
    server.listen(port, "127.0.0.1", () => ok(server));
  });
}

/**
 * A helper page in the given browser that turns PNG buffers into WebP (Chrome's own encoder) and builds the
 * side-by-side composites. Returns { webp(png, quality), composite(leftPng, rightPng, labels), close() }.
 */
export async function imageTools(browser) {
  const ctx = await browser.newContext({ deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.setContent("<!doctype html><body style='margin:0'></body>");
  const encode = (b64, quality, scale = 1) =>
    page.evaluate(
      async ({ b64, quality, scale }) => {
        const img = new Image();
        img.src = `data:image/png;base64,${b64}`;
        await img.decode();
        const c = document.createElement("canvas");
        c.width = Math.round(img.naturalWidth * scale);
        c.height = Math.round(img.naturalHeight * scale);
        const g = c.getContext("2d");
        g.imageSmoothingQuality = "high";
        g.drawImage(img, 0, 0, c.width, c.height);
        return c.toDataURL("image/webp", quality).split(",")[1];
      },
      { b64, quality, scale },
    );
  return {
    async webp(png, quality = 0.86, scale = 1) {
      return Buffer.from(await encode(png.toString("base64"), quality, scale), "base64");
    },
    /** reference | actual, labelled, on one canvas; wide desktop pairs are halved so the file stays small. */
    async composite(left, right, { leftLabel = "reference (prototype)", rightLabel = "actual (app)", title = "" } = {}, quality = 0.82) {
      const b64 = await page.evaluate(
        async ({ l, r, leftLabel, rightLabel, title, quality }) => {
          const load = async (b) => {
            const i = new Image();
            i.src = b ? `data:image/png;base64,${b}` : "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'/>";
            await i.decode();
            return i;
          };
          const [a, b] = await Promise.all([load(l), load(r)]);
          const w = Math.max(a.naturalWidth, b.naturalWidth || 0, 10);
          const h = Math.max(a.naturalHeight, b.naturalHeight || 0, 10);
          const scale = w * 2 > 2000 ? 0.5 : 1;
          const gap = 16, head = 34;
          const c = document.createElement("canvas");
          c.width = Math.round((w * 2 + gap) * scale);
          c.height = Math.round(h * scale) + head;
          const g = c.getContext("2d");
          g.fillStyle = "#1b1f24";
          g.fillRect(0, 0, c.width, c.height);
          g.fillStyle = "#e8edf1";
          g.font = "600 14px system-ui, sans-serif";
          g.textBaseline = "middle";
          g.fillText(`${title}  ·  ${leftLabel}`, 8, head / 2);
          g.fillText(r ? rightLabel : `${rightLabel}: missing`, Math.round((w + gap) * scale) + 8, head / 2);
          g.imageSmoothingQuality = "high";
          g.drawImage(a, 0, head, Math.round(a.naturalWidth * scale), Math.round(a.naturalHeight * scale));
          if (r) g.drawImage(b, Math.round((w + gap) * scale), head, Math.round(b.naturalWidth * scale), Math.round(b.naturalHeight * scale));
          return c.toDataURL("image/webp", quality).split(",")[1];
        },
        { l: left?.toString("base64") ?? null, r: right?.toString("base64") ?? null, leftLabel, rightLabel, title, quality },
      );
      return Buffer.from(b64, "base64");
    },
    /** Decode a WebP (or PNG) file back to PNG bytes, for composites built from saved references. */
    async png(buf, mime = "image/webp") {
      const b64 = await page.evaluate(
        async ({ b64, mime }) => {
          const i = new Image();
          i.src = `data:${mime};base64,${b64}`;
          await i.decode();
          const c = document.createElement("canvas");
          c.width = i.naturalWidth;
          c.height = i.naturalHeight;
          c.getContext("2d").drawImage(i, 0, 0);
          return c.toDataURL("image/png").split(",")[1];
        },
        { b64: buf.toString("base64"), mime },
      );
      return Buffer.from(b64, "base64");
    },
    close: () => ctx.close(),
  };
}

export const IMPL_DIR = "docs/coastwatch/v14/implementation";
