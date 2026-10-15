// Minimal static server for the carto prototype (ES-module MapLibre needs http, not file://).
//   node serve.mjs [port]   then open http://localhost:3399/index.html
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, normalize, extname } from "node:path";

const ROOT = dirname(fileURLToPath(import.meta.url));
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".mjs": "text/javascript",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".geojson": "application/geo+json",
  ".webp": "image/webp",
  ".png": "image/png",
  ".gz": "application/octet-stream",
  ".md": "text/markdown; charset=utf-8",
};

export function startServer(port = 3399) {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, "http://x");
      let p = normalize(decodeURIComponent(url.pathname));
      if (p === "/" || p === "\\") p = "/index.html";
      const file = join(ROOT, p);
      if (!file.startsWith(ROOT)) throw Object.assign(new Error("forbidden"), { code: "EACCES" });
      const s = await stat(file);
      if (!s.isFile()) throw Object.assign(new Error("not a file"), { code: "ENOENT" });
      const body = await readFile(file);
      res.writeHead(200, { "content-type": TYPES[extname(file)] ?? "application/octet-stream", "content-length": body.length, "cache-control": "no-store" });
      res.end(body);
    } catch (e) {
      res.writeHead(e.code === "ENOENT" ? 404 : 500, { "content-type": "text/plain" });
      res.end(String(e.message));
    }
  });
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve(server)));
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const port = +(process.argv[2] ?? 3399);
  await startServer(port);
  console.log(`carto prototype at http://localhost:${port}/index.html`);
}
