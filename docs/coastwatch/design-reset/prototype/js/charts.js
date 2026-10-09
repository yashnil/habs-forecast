/* Minimal SVG chart helpers (no dependencies). Scales never depend on hidden data:
   probability axes are always 0-100 %, log axes span whole decades. */
(function () {
  const lin = (d0, d1, r0, r1) => {
    const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
    f.domain = [d0, d1];
    return f;
  };
  const log = (d0, d1, r0, r1) => {
    const a = Math.log10(d0), b = Math.log10(d1);
    const f = (v) => r0 + ((Math.log10(v) - a) / (b - a)) * (r1 - r0);
    f.domain = [d0, d1];
    return f;
  };

  const MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  /** Month or year ticks between two timestamps, at most ~maxTicks. */
  function timeTicks(t0, t1, maxTicks = 8) {
    const spanM = (t1 - t0) / (30.44 * 86400000);
    const ticks = [];
    if (spanM > maxTicks * 6) {
      const step = Math.ceil(spanM / 12 / maxTicks);
      for (let y = new Date(t0).getUTCFullYear() + 1; Date.UTC(y, 0, 1) <= t1; y += step) ticks.push({ t: Date.UTC(y, 0, 1), label: String(y), major: true });
      return ticks;
    }
    const step = [1, 2, 3, 6].find((s) => spanM / s <= maxTicks) ?? 12;
    const d = new Date(t0);
    let y = d.getUTCFullYear(), m = d.getUTCMonth() + 1;
    for (;;) {
      if (m > 11) { y += 1; m -= 12; }
      const t = Date.UTC(y, m, 1);
      if (t > t1) break;
      if (m % step === 0) ticks.push({ t, label: m === 0 ? String(y) : MONTH[m], major: m === 0 });
      m += 1;
    }
    return ticks;
  }

  /** Probability display classes, 10 % wide, matching the map raster (not risk levels). */
  const P_CLASSES = ["#3a385b", "#4c436a", "#5f4e79", "#735986", "#886492", "#9c709c", "#af7ea4", "#c28cab", "#d39cb3", "#e5abbc"];
  const pClass = (v) => P_CLASSES[Math.max(0, Math.min(9, Math.floor(v * 10)))];

  function sparkline(points, { w = 280, h = 56, color = "#6a3fb0", t0, t1, area = true, yLabel = true } = {}) {
    if (!points.length) return "";
    const x = lin(t0 ?? points[0].t, t1 ?? points.at(-1).t, 2, w - 2);
    const y = lin(0, 1, h - 4, 4);
    const segs = [];
    let cur = [];
    points.forEach((p, i) => {
      if (i && p.t - points[i - 1].t > 2.5 * 86400000) { segs.push(cur); cur = []; }
      cur.push(p);
    });
    segs.push(cur);
    const path = (s) => s.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join("");
    const fill = area ? segs.filter((s) => s.length > 1).map((s) => `<path d="${path(s)}L${x(s.at(-1).t).toFixed(1)},${h - 4}L${x(s[0].t).toFixed(1)},${h - 4}Z" fill="${color}" opacity=".10"/>`).join("") : "";
    return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img">
      <line x1="0" x2="${w}" y1="${y(0.5)}" y2="${y(0.5)}" stroke="#e4e0d7" stroke-dasharray="2 3"/>
      <line x1="0" x2="${w}" y1="${h - 4}" y2="${h - 4}" stroke="#e4e0d7"/>
      ${fill}${segs.map((s) => `<path d="${path(s)}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round"/>`).join("")}
      ${points.length ? `<circle cx="${x(points.at(-1).t)}" cy="${y(points.at(-1).v)}" r="2.8" fill="${color}"/>` : ""}
      ${yLabel ? `<text x="${w}" y="${y(0.5) - 3}" text-anchor="end" style="font:400 10px var(--font-sans);fill:#5f6b79">50%</text>` : ""}
    </svg>`;
  }

  window.CWChart = { lin, log, timeTicks, MONTH, P_CLASSES, pClass, sparkline };
})();
