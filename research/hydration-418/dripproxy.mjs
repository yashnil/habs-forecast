// proxy: HTML documents are sent in small chunks with pauses (like a slow network); everything else passes straight through
import http from "node:http";
const [port, target, chunk = "4096", pause = "25"] = process.argv.slice(2);
http.createServer((req, res) => {
  const up = http.request({ host: "localhost", port: +target, path: req.url, method: req.method, headers: { ...req.headers, host: `localhost:${target}`, "accept-encoding": "identity" } }, async (r) => {
    const html = /text\/html/.test(r.headers["content-type"] ?? "");
    if (!html) { res.writeHead(r.statusCode, r.headers); r.pipe(res); return; }
    const h = { ...r.headers }; delete h["content-length"];
    res.writeHead(r.statusCode, h);
    const parts = []; for await (const c of r) parts.push(c);
    const buf = Buffer.concat(parts);
    for (let i = 0; i < buf.length; i += +chunk) { res.write(buf.subarray(i, i + +chunk)); await new Promise((ok) => setTimeout(ok, +pause)); }
    res.end();
  });
  up.on("error", () => { res.writeHead(502); res.end(); });
  req.pipe(up);
}).listen(+port);
