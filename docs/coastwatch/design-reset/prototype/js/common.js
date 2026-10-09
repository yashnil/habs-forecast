/* Shared prototype runtime: formatting, freshness, official notices, masthead, drawer, tab bar.
   Every value shown comes from window.CW_DATA (a snapshot of the production Pages dataset). */
(function () {
  const D = window.CW_DATA;
  const TZ = "America/Los_Angeles";
  // "Now" is the snapshot time, so ages in the prototypes match the published run.
  const NOW = Date.parse(D.snapshot.generated_at);
  const DAY = 86400000;

  const REGIONS = [
    { id: "north_coast", label: "North Coast", bounds: [[-124.75, 39.9], [-123.6, 42.05]] },
    { id: "mendocino_sonoma", label: "Mendocino–Sonoma", bounds: [[-124.0, 38.2], [-122.8, 39.95]] },
    { id: "sf_bay_farallones", label: "San Francisco & Farallones", bounds: [[-123.35, 37.25], [-122.2, 38.15]] },
    { id: "monterey_bay", label: "Monterey Bay", bounds: [[-122.45, 36.45], [-121.72, 37.15]] },
    { id: "central_coast", label: "Central Coast", bounds: [[-121.6, 34.4], [-120.4, 36.1]] },
    { id: "southern_california", label: "Southern California", bounds: [[-120.6, 32.45], [-117.05, 34.55]] },
  ];
  const REGION = Object.fromEntries(REGIONS.map((r) => [r.id, r]));
  const CALIFORNIA = [[-125.2, 32.3], [-116.9, 42.1]];

  const ACTION = {
    quarantine: "Quarantine",
    consumption_advisory: "Health advisory",
    take_restriction: "Take restriction",
    fishery_closure: "Fishery closure",
    special_advisory: "Special advisory",
  };
  const FISHERY = { sport_harvest: "Sport harvest", consumption: "Eating / consumption", commercial: "Commercial", recreational: "Recreational", commercial_and_recreational: "Commercial and recreational" };

  const ICON = {
    map: '<path d="M9 4 3 6.5v13.5l6-2.5 6 2.5 6-2.5V4l-6 2.5L9 4Z"/><path d="M9 4v13.5M15 6.5V20"/>',
    bloom: '<circle cx="6" cy="17" r="2.2"/><circle cx="12" cy="12" r="2.2"/><circle cx="18" cy="7" r="2.2"/><path d="m7.6 15.4 2.8-1.8M13.6 10.4l2.8-1.8"/>',
    fish: '<path d="M3 12c3-4.5 7.5-6 11-4.5 2.2 1 3.8 2.7 4.8 4.5-1 1.8-2.6 3.5-4.8 4.5C10.5 18 6 16.5 3 12Z"/><path d="m18.8 12 2.2-3v6l-2.2-3Z"/><circle cx="14" cy="11" r=".6" fill="currentColor"/>',
    shield: '<path d="M12 3 4.5 6v5.5c0 4.5 3.2 8 7.5 9.5 4.3-1.5 7.5-5 7.5-9.5V6L12 3Z"/><path d="M12 8v5M12 16.2v.1"/>',
    close: '<path d="M6 6l12 12M18 6 6 18"/>',
    chevron: '<path d="m9 6 6 6-6 6"/>',
    down: '<path d="m6 9 6 6 6-6"/>',
    ext: '<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
    search: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>',
    info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8v.1"/>',
    lock: '<rect x="5" y="10.5" width="14" height="9.5" rx="2"/><path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5"/>',
    layers: '<path d="m12 4 8.5 4.5L12 13 3.5 8.5 12 4Z"/><path d="m3.5 12.5 8.5 4.5 8.5-4.5M3.5 16.5 12 21l8.5-4.5"/>',
    phone: '<path d="M5 4h4l2 5-2.5 1.5a11 11 0 0 0 5 5L15 13l5 2v4a1 1 0 0 1-1 1A16 16 0 0 1 4 5a1 1 0 0 1 1-1Z"/>',
    pin: '<path d="M12 21s-6.5-6-6.5-11a6.5 6.5 0 0 1 13 0c0 5-6.5 11-6.5 11Z"/><circle cx="12" cy="10" r="2.2"/>',
  };
  const icon = (name, cls = "") => `<svg class="icon ${cls}" viewBox="0 0 24 24" aria-hidden="true">${ICON[name]}</svg>`;
  const LOGO = `<svg viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="8" fill="#10263e"/><path d="M6 19c3.2-2.6 6.4-2.6 9.6 0s6.6 2.6 10.4-.2" fill="none" stroke="#6cc6dc" stroke-width="2.2" stroke-linecap="round"/><path d="M6 13.5c3.2-2.6 6.4-2.6 9.6 0s6.6 2.6 10.4-.2" fill="none" stroke="#e8eef5" stroke-opacity=".55" stroke-width="2.2" stroke-linecap="round"/><circle cx="22.5" cy="8.5" r="2.2" fill="#f6bb5c"/></svg>`;

  // ---------- dates ----------
  const isDateOnly = (s) => /^\d{4}-\d{2}-\d{2}$/.test(s);
  function fmtDate(s, opts = { month: "short", day: "numeric", year: "numeric" }) {
    if (s == null) return "—";
    const d = typeof s === "number" ? new Date(s) : new Date(isDateOnly(s) ? s + "T12:00:00Z" : s);
    return new Intl.DateTimeFormat("en-US", { timeZone: isDateOnly(String(s)) ? "UTC" : TZ, ...opts }).format(d);
  }
  const fmtDay = (s) => fmtDate(s, { weekday: "short", month: "short", day: "numeric" });
  const fmtTime = (s) => fmtDate(s, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" });
  function pacificDay(ms) {
    return new Intl.DateTimeFormat("en-CA", { timeZone: TZ, year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date(ms));
  }
  // Whole calendar days between a date (or timestamp) and "now", in Pacific time.
  function daysAgo(s) {
    const d = isDateOnly(s) ? s : pacificDay(Date.parse(s));
    return Math.round((Date.parse(pacificDay(NOW)) - Date.parse(d)) / DAY);
  }
  const agoText = (n) => (n <= 0 ? "today" : n === 1 ? "yesterday" : `${n} days ago`);

  function freshness(dateStr, policy) {
    const n = daysAgo(dateStr);
    const state = n <= policy.current ? "current" : n <= policy.stale ? "stale" : "historical";
    return { state, days: n, label: { current: "Current", stale: "Stale", historical: "Historical — not current" }[state] };
  }
  const freshChip = (f, text) => `<span class="fresh" data-state="${f.state}">${text ?? f.label}</span>`;

  // ---------- numbers ----------
  const pct = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);
  function money(v, digits) {
    if (v == null) return "—";
    const a = Math.abs(v);
    if (a >= 1e9) return `$${(v / 1e9).toFixed(digits ?? 1)}B`;
    if (a >= 1e6) return `$${(v / 1e6).toFixed(digits ?? (a >= 1e7 ? 0 : 1))}M`;
    if (a >= 1e3) return `$${(v / 1e3).toFixed(digits ?? 0)}k`;
    return `$${Math.round(v)}`;
  }
  function sig(v, n = 3) {
    if (v == null) return "—";
    if (v === 0) return "0";
    const d = Math.max(0, n - 1 - Math.floor(Math.log10(Math.abs(v))));
    return Number(v.toFixed(d)).toLocaleString("en-US", { maximumFractionDigits: d });
  }
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  // ---------- official ----------
  const OFF = D.official;
  const records = OFF.registry.records.filter((r) => r.status === "active");
  const recordById = Object.fromEntries(OFF.registry.records.map((r) => [r.id, r]));
  const review = OFF.registry.review;
  const verified = review.status === "verified";
  const portById = Object.fromEntries(D.port_intel.ports.map((p) => [p.port_code, p]));
  function noticesForPort(code) {
    const p = portById[code];
    return p ? p.official_relations.map((rel) => ({ record: recordById[rel.record_id], rel })).filter((x) => x.record && x.record.status === "active") : [];
  }
  function noticesForRegion(region) {
    const seen = new Map();
    for (const p of D.port_intel.ports.filter((p) => p.region === region)) for (const x of noticesForPort(p.port_code)) seen.set(x.record.id, x);
    return [...seen.values()];
  }
  const agencyChip = (a) => `<span class="chip chip-agency">${esc(a)}</span>`;
  const shortTitle = (r) => r.title;

  function verificationLine() {
    return verified
      ? `<span class="unverified">Reviewed ${fmtDate(review.reviewed_at)}</span>`
      : `<span class="unverified">${icon("info", "icon-s")}Not verified</span>`;
  }

  // ---------- masthead, drawer, tab bar ----------
  const PAGES = [
    { id: "map", href: "map.html", label: "Ocean Map", icon: "map" },
    { id: "bloom", href: "bloom.html", label: "Bloom Intelligence", icon: "bloom" },
    { id: "fisheries", href: "fisheries.html", label: "Fisheries", icon: "fish" },
  ];
  function mountChrome(active) {
    const head = document.createElement("header");
    head.className = "masthead";
    head.innerHTML = `
      <a class="brand" href="map.html">${LOGO}<span class="brand-name">CoastWatch</span><span class="brand-region">California</span></a>
      <nav class="nav" aria-label="Primary">${PAGES.map((p) => `<a href="${p.href}" ${p.id === active ? 'aria-current="page"' : ""}>${p.label}</a>`).join("")}</nav>
      <div class="masthead-tools">
        <button class="official-pill" data-open-official aria-haspopup="dialog">${icon("shield", "icon-s")}<b>${records.length}</b><span class="long">official notices</span><span class="sep"></span>${verified ? "Reviewed" : "Not verified"}</button>
        <a class="data-link" href="#sources"><span class="dot"></span>Data status</a>
      </div>`;
    document.body.prepend(head);

    const tab = document.createElement("nav");
    tab.className = "tabbar";
    tab.setAttribute("aria-label", "Primary");
    tab.innerHTML =
      PAGES.map((p) => `<a href="${p.href}" ${p.id === active ? 'aria-current="page"' : ""}>${icon(p.icon)}${p.label.replace("Bloom Intelligence", "Blooms").replace("Ocean Map", "Map")}</a>`).join("") +
      `<a href="#notices" class="notices" data-open-official>${icon("shield")}<span class="badge">${records.length}</span>Notices</a>`;
    document.body.append(tab);
    mountDrawer();
    document.addEventListener("click", (e) => {
      if (e.target.closest("[data-open-official]")) { e.preventDefault(); openDrawer(e.target.closest("[data-open-official]").dataset.focus); }
      if (e.target.closest("[data-close-official]") || e.target.classList.contains("scrim")) document.body.classList.remove("drawer-open");
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") document.body.classList.remove("drawer-open"); });
  }

  function noticeBlock(r) {
    const dates = [r.effective_date ? `Since ${fmtDate(r.effective_date)}` : r.effective_date_note ? "Start date not published" : null, r.expected_end_date ? `through at least ${fmtDate(r.expected_end_date)}` : null].filter(Boolean).join(" · ");
    return `<article class="notice" id="notice-${r.id}">
      ${agencyChip(r.agency)}
      <p class="kind">${ACTION[r.action] ?? r.action} · ${FISHERY[r.fishery] ?? r.fishery}</p>
      <h3>${esc(r.title)}</h3>
      <p class="meta">${esc(r.area.description)}${dates ? " · " + dates : ""}</p>
      <blockquote>“${esc(r.official_text)}”</blockquote>
      <p class="src">${r.sources.filter((s) => s.url.startsWith("http")).slice(0, 1).map((s) => `<a href="${s.url}">${esc(s.label)}</a>`).join("")}</p>
    </article>`;
  }
  function mountDrawer() {
    const scrim = document.createElement("div");
    scrim.className = "scrim";
    const d = document.createElement("aside");
    d.className = "drawer";
    d.setAttribute("role", "dialog");
    d.setAttribute("aria-label", "Official closures and advisories");
    const byAgency = ["CDFW", "CDPH"].map((a) => ({ a, rs: records.filter((r) => r.agency === a) }));
    d.innerHTML = `
      <div class="drawer-head">
        <p class="eyebrow official">Official closures and advisories</p>
        <h2>${records.length} active notices in California</h2>
        <div class="verify-note">${icon("info")}<div><b>${verified ? "Reviewed" : "Not verified."}</b> ${verified ? "" : "Transcribed from CDFW and CDPH pages on " + fmtDate(review.reviewed_at) + " and awaiting a person's check against each source. "}The agency pages are the authority. A notice missing here does not mean an area is open or that seafood is safe.</div></div>
        <button class="drawer-close" data-close-official aria-label="Close">${icon("close")}</button>
      </div>
      <div class="drawer-body">
        ${byAgency.map(({ a, rs }) => `<h3 class="eyebrow" style="margin-top:20px">${a === "CDFW" ? "California Department of Fish and Wildlife" : "California Department of Public Health"} · ${rs.length}</h3>${rs.map(noticeBlock).join("")}`).join("")}
        <h3 class="eyebrow" style="margin-top:24px">Agency statements (quoted)</h3>
        ${OFF.registry.statements.map((s) => `<p class="statement"><b>${esc(s.topic)}</b>“${esc(s.statement)}” — ${esc(s.agency)}</p>`).join("")}
        <div class="hotlines">${OFF.registry.hotlines.map((h) => `<div class="hotline"><span>${esc(h.label)}</span><a href="tel:${h.tel}">${h.phone}</a></div>`).join("")}</div>
      </div>`;
    document.body.append(scrim, d);
  }
  function openDrawer(focusId) {
    document.body.classList.add("drawer-open");
    if (focusId) document.getElementById("notice-" + focusId)?.scrollIntoView({ block: "start" });
  }

  // Compact strip used at the top of the analysis pages.
  function officialStrip(items, context) {
    return `<div class="official-strip" data-testid="official-strip">
      <span class="lead">${icon("shield")}Official notices come first</span>
      <div class="items">${items
        .map(({ record: r }) => `<button class="item" data-open-official data-focus="${r.id}">${agencyChip(r.agency)}<span>${esc(r.title)}</span></button>`)
        .join("")}</div>
      <div class="tail">${verificationLine()}<button class="link-arrow" data-open-official>All ${records.length} notices${icon("chevron", "icon-s")}</button></div>
    </div>`;
  }

  window.CW = {
    D, NOW, DAY, TZ, REGIONS, REGION, CALIFORNIA, ACTION, FISHERY, icon, LOGO,
    fmtDate, fmtDay, fmtTime, daysAgo, agoText, pacificDay, freshness, freshChip,
    pct, money, sig, esc,
    records, recordById, review, verified, portById, noticesForPort, noticesForRegion, agencyChip, shortTitle, verificationLine,
    mountChrome, openDrawer, officialStrip,
  };
})();
