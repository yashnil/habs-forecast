/* Fisheries & Economic Exposure prototype. URL params: ?tiers=1|12&dollars=real|nominal&group=<id> */
(function () {
  const { D, icon, esc, money, fmtDate } = CW;
  const { lin } = CWChart;
  CW.mountChrome("fisheries");

  const F = D.fisheries;
  const COLOR = { dungeness_crab: "var(--sp-dungeness)", rock_crab: "var(--sp-rockcrab)", northern_anchovy: "var(--sp-anchovy)", spiny_lobster: "var(--sp-lobster)", pacific_sardine: "var(--sp-sardine)", bivalves: "var(--sp-bivalves)" };
  const SHORT = { bivalves: "Bivalve shellfish", spiny_lobster: "Spiny lobster" };
  const label = (g) => SHORT[g.id] ?? g.label;
  const q = new URLSearchParams(location.search);
  const state = {
    tiers: q.get("tiers") === "12" ? "12" : "1",
    dollars: q.get("dollars") === "nominal" ? "nominal" : "real",
    group: F.groups.some((g) => g.id === q.get("group")) ? q.get("group") : null,
  };
  const years = F.years;
  const latest = years.at(-1);
  const key = () => (state.dollars === "real" ? "dollars_real" : "dollars_nominal");
  const unitText = () => (state.dollars === "real" ? `${F.deflator.base_year} dollars (CPI-U adjusted)` : "nominal dollars, as reported");
  const groups = () => F.groups.filter((g) => (state.tiers === "1" ? g.tier === 1 : true));
  const val = (g, y) => g.annual.find((a) => a.year === y)?.[key()] ?? null;
  const total = (gs, y) => gs.reduce((s, g) => s + (val(g, y) ?? 0), 0);
  const state_total = (y) => F.statewide_total.find((t) => t.year === y)?.[key()] ?? null;
  const withheld = (y) => F.withheld.find((t) => t.year === y)?.[key()] ?? null;
  const mobile = () => matchMedia("(max-width: 720px)").matches;

  // ---------- official strip: notices tied to these species groups ----------
  function renderStrip() {
    const ids = [...new Set(F.groups.flatMap((g) => g.official_record_ids))];
    const items = ids.map((id) => ({ record: CW.recordById[id] })).filter((x) => x.record?.status === "active");
    document.getElementById("strip").innerHTML = CW.officialStrip(items) + `
      <button class="m-strip" data-open-official>${icon("shield")}<span><b>${items.length} official notices</b> concern these species</span><span class="unverified">Not verified</span>${icon("chevron", "icon-s")}</button>`;
  }

  // ---------- primary chart: stacked annual bars ----------
  function barChart(w, h) {
    const gs = groups();
    const totals = years.map((y) => total(gs, y));
    const maxV = Math.max(...totals);
    const step = maxV > 100e6 ? 40e6 : maxV > 50e6 ? 20e6 : 10e6;
    const top = Math.ceil(maxV / step) * step;
    const m = { l: mobile() ? 52 : 64, r: 4, t: mobile() ? 28 : 48, b: 58 };
    const x0 = m.l, x1 = w - m.r, slot = (x1 - x0) / years.length, bw = slot * (mobile() ? 0.7 : 0.62);
    const y = lin(0, top, h - m.b, m.t);
    const ticks = Array.from({ length: Math.round(top / step) + 1 }, (_, i) => i * step);
    const cx = (i) => x0 + slot * i + slot / 2;
    const sel = state.group;
    let bars = "";
    years.forEach((yr, i) => {
      let acc = 0;
      for (const g of gs) {
        const v = val(g, yr) ?? 0;
        const y0 = y(acc), y1 = y(acc + v);
        acc += v;
        const dim = sel && sel !== g.id;
        bars += `<rect x="${cx(i) - bw / 2}" y="${y1}" width="${bw}" height="${Math.max(0, y0 - y1 - 0.5)}" fill="${COLOR[g.id]}" opacity="${dim ? 0.18 : 1}"><title>${label(g)} ${yr}: ${money(v)}</title></rect>`;
      }
      const shown = sel ? val(gs.find((g) => g.id === sel) ?? F.groups.find((g) => g.id === sel), yr) : acc;
      bars += `<text x="${cx(i)}" y="${y(acc) - 8}" text-anchor="middle" class="bar-total">${money(shown, shown >= 1e7 ? 0 : 1)}</text>`;
      const share = state_total(yr) ? acc / state_total(yr) : null;
      bars += `<text x="${cx(i)}" y="${h - m.b + 18}" text-anchor="middle" class="yr ${yr === latest ? "cur" : ""}">${mobile() ? "’" + String(yr).slice(2) : yr}</text>`;
      bars += `<text x="${cx(i)}" y="${h - m.b + 38}" text-anchor="middle" class="share">${share != null ? Math.round(share * 100) + "%" : "—"}</text>`;
    });
    // 2015–16: the tier-basis text records the domoic-acid delay of that Dungeness season.
    const ann = `<g class="ann"><path d="M${cx(0) - bw / 2},${m.t - 14}H${cx(1) + bw / 2}" stroke="var(--ink-2)" fill="none"/><path d="M${cx(0) - bw / 2},${m.t - 14}v5M${cx(1) + bw / 2},${m.t - 14}v5" stroke="var(--ink-2)"/><text x="${cx(0) - bw / 2}" y="${m.t - 22}" class="ann-t">2015–16 Dungeness season delayed by domoic acid (CDFW)</text></g>`;
    return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="Landed value by year, stacked by species group">
      <g class="grid">${ticks.map((t) => `<line x1="${x0}" x2="${x1}" y1="${y(t)}" y2="${y(t)}"/>`).join("")}</g>
      ${ticks.map((t) => `<text x="${x0 - 8}" y="${y(t) + 4}" text-anchor="end">${money(t)}</text>`).join("")}
      ${bars}${state.tiers && !mobile() ? ann : ""}
      <text x="${x0 - 8}" y="${h - m.b + 38}" text-anchor="end" class="share-l">${mobile() ? "" : "share"}</text>
    </svg>`;
  }

  function sparkBars(g, w = 120, h = 30) {
    const vals = years.map((y) => val(g, y) ?? 0);
    const mx = Math.max(...vals) || 1;
    const bw = w / years.length;
    return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">${vals.map((v, i) => `<rect x="${i * bw + 1}" y="${h - (v / mx) * h}" width="${bw - 2}" height="${(v / mx) * h}" rx="1" fill="${COLOR[g.id]}" opacity="${i === vals.length - 1 ? 1 : 0.55}"/>`).join("")}</svg>`;
  }

  function detail(g) {
    const a = g.annual.find((x) => x.year === latest);
    const avg = years.reduce((s, y) => s + (val(g, y) ?? 0), 0) / years.length;
    const recs = g.official_record_ids.map((id) => CW.recordById[id]).filter(Boolean);
    const extra = g.id === "bivalves" ? `<p class="d-note">${esc(F.caveats.find((c) => c.startsWith("Bivalve")) ?? "")}</p>` : "";
    return `<div class="detail" data-testid="group-detail">
      <p class="eyebrow">Tier ${g.tier} · ${g.tier === 1 ? "toxin closures" : "advisories"}</p>
      <h3>${esc(g.label)}</h3>
      <div class="d-figs">
        <div><span class="d-n">${money(val(g, latest))}</span><span class="fine">in ${latest}${state.dollars === "real" ? ` (${F.deflator.base_year} $)` : " (nominal)"}</span></div>
        <div><span class="d-n">${money(avg)}</span><span class="fine">average ${years[0]}–${latest}</span></div>
        <div><span class="d-n">${a?.pounds != null ? (a.pounds / 1e6).toFixed(1) + "M lb" : "—"}</span><span class="fine">landed in ${latest}${g.id === "bivalves" ? " (meat weight)" : ""}</span></div>
      </div>
      <p class="d-basis">${esc(g.tier_basis)}</p>
      ${recs.length ? `<div class="d-off">${recs.map((r) => `<button class="notice-row" data-open-official data-focus="${r.id}">${CW.agencyChip(r.agency)}<span class="t" style="white-space:normal">${esc(r.title)}</span>${icon("chevron", "icon-s")}</button>`).join("")}</div>` : `<p class="fine">No active official notice is linked to this group.</p>`}
      ${extra}
      <p class="fine">NOAA market categories: ${g.source_names.map(esc).join(", ")}</p>
    </div>`;
  }

  function render() {
    const gs = groups();
    const main = document.getElementById("main");
    const W = Math.min(main.clientWidth, 1120) - (mobile() ? 32 : 0);
    const tLatest = total(gs, latest);
    const avg = years.reduce((s, y) => s + total(gs, y), 0) / years.length;
    const totals = years.map((y) => total(gs, y));
    const minY = years[totals.indexOf(Math.min(...totals))], maxY = years[totals.indexOf(Math.max(...totals))];
    const share = tLatest / state_total(latest);
    const sel = F.groups.find((g) => g.id === state.group) ?? null;
    const areas = [...new Set(D.ports.features.slice().sort((a, b) => b.geometry.coordinates[1] - a.geometry.coordinates[1]).map((f) => f.properties.port_area))];
    const title = (s) => s.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
    const dup = F.excluded_rows;

    main.innerHTML = `
      <header class="hero">
        <div class="hero-text">
          <p class="eyebrow"><span class="chip chip-history">Historical</span> California, statewide · ${years[0]}–${latest} · NOAA Fisheries landings</p>
          <h1>What California’s toxin-affected fisheries have landed</h1>
          <p class="lede" data-testid="exposure-definition">${esc(F.terminology)}</p>
        </div>
        <p class="through" data-testid="data-through">Annual data through <b>${latest}</b>. ${F.years_requested_unavailable.join(", ")} not yet published by NOAA.</p>
      </header>

      <div class="controls" role="group" aria-label="Chart options">
        <div class="ctl"><span class="ctl-l">Species</span><div class="seg"><button data-tiers="1" aria-pressed="${state.tiers === "1"}">Tier 1<span class="long"> · toxin closures</span></button><button data-tiers="12" aria-pressed="${state.tiers === "12"}">Tiers 1 + 2<span class="long"> · with advisories</span></button></div></div>
        <div class="ctl"><span class="ctl-l">Dollars</span><div class="seg"><button data-dollars="real" aria-pressed="${state.dollars === "real"}">${F.deflator.base_year} dollars</button><button data-dollars="nominal" aria-pressed="${state.dollars === "nominal"}">Nominal</button></div></div>
      </div>

      <section class="figures" data-testid="fisheries-tiles" aria-label="Key figures">
        <div class="fig"><span class="fig-n">${money(tLatest)}</span><span class="fig-l">landed value in ${latest}</span><span class="fine">${gs.length} species groups · ${unitText()}</span></div>
        <div class="fig"><span class="fig-n">${money(avg)}</span><span class="fig-l">average per year, ${years[0]}–${latest}</span><span class="fine">lowest ${money(Math.min(...totals))} (${minY}) · highest ${money(Math.max(...totals))} (${maxY})</span></div>
        <div class="fig"><span class="fig-n">${Math.round(share * 100)}%</span><span class="fig-l">of all California commercial landings, ${latest}</span><span class="fine">NOAA state total ${money(state_total(latest))}, which includes ${money(withheld(latest))} withheld as confidential</span></div>
      </section>
      <p class="not-loss">${icon("info", "icon-s")}Past landings show what was at stake in earlier seasons. They are not losses, not a forecast of losses, and say nothing about the current season.</p>

      <section class="card primary" aria-labelledby="pc-title">
        <div class="pc-head">
          <div><h2 id="pc-title">Landed value by year${sel ? ` · <span style="color:${COLOR[sel.id]}">${esc(label(sel))}</span>` : ""}</h2><p class="fine">${unitText()}. Past landings only, not a forecast. Bottom row: share of NOAA’s state total.</p></div>
          <div class="legend">${gs.map((g) => `<button class="k ${state.group === g.id ? "on" : ""}" data-group="${g.id}"><i class="swatch" style="background:${COLOR[g.id]}"></i>${esc(label(g))}</button>`).join("")}</div>
        </div>
        <div class="chart">${barChart(mobile() ? W : W - 56, mobile() ? 300 : 380)}</div>
      </section>

      <section id="breakdown" class="breakdown" aria-labelledby="bd-title">
        <div class="section-head"><div><h2 id="bd-title">By species group</h2><p>Select a group to highlight it in the chart.</p></div></div>
        <div class="bd-grid">
          <table class="bd-table">
            <thead><tr><th>Group</th><th class="num">${latest}</th><th class="num hide-m">10-yr avg</th><th class="hide-m">${years[0]}–${latest}</th><th class="num">Share</th></tr></thead>
            <tbody>${F.groups
              .map((g) => {
                const inSet = gs.includes(g);
                const gAvg = years.reduce((s, y) => s + (val(g, y) ?? 0), 0) / years.length;
                return `<tr class="${inSet ? "" : "out"} ${state.group === g.id ? "on" : ""}" data-group="${g.id}" tabindex="0">
                  <td><span class="g"><i class="swatch" style="background:${COLOR[g.id]}"></i><span>${esc(label(g))}<small>Tier ${g.tier}</small></span></span></td>
                  <td class="num"><b>${money(val(g, latest))}</b></td>
                  <td class="num hide-m">${money(gAvg)}</td>
                  <td class="hide-m">${sparkBars(g)}</td>
                  <td class="num">${inSet ? Math.round(((val(g, latest) ?? 0) / tLatest) * 100) + "%" : "—"}</td>
                </tr>`;
              })
              .join("")}</tbody>
          </table>
          <aside class="bd-detail">${sel ? detail(sel) : `<div class="detail empty"><p class="eyebrow">Species detail</p><p>Select a group to see why it is included, which official notices concern it, and which NOAA categories it combines.</p><p class="fine">Tiers are CoastWatch’s editorial grouping and await review by an independent HAB scientist.</p></div>`}</aside>
        </div>
        <details class="pna" data-testid="port-level-unavailable">
          <summary>${icon("lock", "icon-s")}<span><b>Port-level values are not available yet.</b> Everything here is statewide and is never divided among ports.</span><span class="pna-why">Why</span></summary>
          <div class="pna-body">
            <ul>${F.port_level.reasons.filter((r) => !r.startsWith("Statewide values below")).map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
            <p class="fine">CDFW port areas, north to south: ${areas.map(title).join(" · ")}.</p>
          </div>
        </details>
        <p class="fine bd-note">Groups leave out generic categories (for example unspecified crabs), so they are lower bounds. Landings withheld for confidentiality (${money(withheld(latest))} in ${latest}) are never attributed to a group.</p>
      </section>


      <section class="methods" aria-labelledby="m-title">
        <h2 id="m-title" class="section-title">How these numbers are made</h2>
        <div class="m-grid">
          <div><p class="eyebrow">Source</p><p>NOAA Fisheries FOSS commercial landings for California (from PacFIN), one request per year. Courtesy: National Oceanic and Atmospheric Administration.</p></div>
          <div data-testid="deflator"><p class="eyebrow">Inflation</p><p>CPI-U (BLS ${F.deflator.series_id}), base year ${F.deflator.base_year}: the latest year with all 12 monthly values. 2025 has ${F.deflator.months_used["2025"]} of 12 (missing M10), so it cannot be the base.</p></div>
          <div><p class="eyebrow">Confidentiality</p><p>NOAA combines confidential landings into one withheld value per year. It counts toward the state total, as in NOAA’s own totals, and is never assigned to a species.</p></div>
          <div><p class="eyebrow">Duplicates</p><p>${dup.length} upstream rows repeat another row exactly (${esc(dup[0]?.source_name.replace(" **", ""))} = ${esc(dup[0]?.duplicate_of)}) and are counted once.</p></div>
        </div>
        <details class="disclosure"><summary>All caveats</summary><div class="body"><ul>${F.caveats.map((c) => `<li>${esc(c)}</li>`).join("")}</ul></div></details>
        <details class="disclosure"><summary>Values table, all years</summary><div class="body tbl-wrap"><table class="vt"><thead><tr><th>Group</th>${years.map((y) => `<th class="num">${y}</th>`).join("")}</tr></thead><tbody>
          ${F.groups.map((g) => `<tr><td>${esc(label(g))}</td>${years.map((y) => `<td class="num">${money(val(g, y))}</td>`).join("")}</tr>`).join("")}
          <tr class="tot"><td>NOAA state total</td>${years.map((y) => `<td class="num">${money(state_total(y))}</td>`).join("")}</tr>
          <tr data-testid="withheld-row"><td>Withheld (not attributed)</td>${years.map((y) => `<td class="num">${money(withheld(y))}</td>`).join("")}</tr>
        </tbody></table></div></details>
        <details class="disclosure"><summary>Processing steps</summary><div class="body"><ul>${Object.values(F.method).map((m) => `<li>${esc(m)}</li>`).join("")}</ul></div></details>
        <p class="fine source">${esc(F.provenance[0].source_name)} · retrieved ${CW.fmtTime(F.provenance[0].retrieved_at ?? F.generated_at)}. ${esc(F.provenance[0].license)}</p>
      </section>`;
  }

  function sync() {
    const u = new URLSearchParams();
    if (state.tiers !== "1") u.set("tiers", state.tiers);
    if (state.dollars !== "real") u.set("dollars", state.dollars);
    if (state.group) u.set("group", state.group);
    history.replaceState(null, "", "?" + u.toString());
  }
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-tiers],[data-dollars],[data-group]");
    if (!t) return;
    if (t.dataset.tiers) state.tiers = t.dataset.tiers;
    else if (t.dataset.dollars) state.dollars = t.dataset.dollars;
    else if (t.dataset.group) state.group = state.group === t.dataset.group ? null : t.dataset.group;
    if (state.group && state.tiers === "1" && F.groups.find((g) => g.id === state.group).tier === 2) state.tiers = "12";
    render(); sync();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Enter" && e.target.matches("tr[data-group]")) e.target.click(); });
  renderStrip();
  render();
  let rw = innerWidth;
  addEventListener("resize", () => { if (innerWidth !== rw) { rw = innerWidth; render(); } });
})();
