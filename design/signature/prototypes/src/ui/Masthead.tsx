import { useState } from "react";

/**
 * Masthead: one navy bar. The official-status line is part of it (not a second cream bar): the
 * count of notices in effect, their verification state, and the two agencies to check — always
 * visible, one line, never dismissible. Primary navigation keeps the redesign's IA.
 */
export function Masthead({ mode, onMode, notices, unverified, onNotices, compact }: {
  mode: "coastal" | "research";
  onMode: (m: "coastal" | "research") => void;
  notices: number | null;
  unverified: boolean;
  onNotices: () => void;
  compact?: boolean;
}) {
  const [menu, setMenu] = useState(false);
  return (
    <header className="mh" data-compact={compact || undefined}>
      <a className="mh-brand" href="#/" aria-label="CoastWatch prototypes index">
        <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
          <circle cx="11" cy="11" r="10.25" fill="none" stroke="currentColor" strokeOpacity=".35" />
          <path d="M3.5 12.5c2.2-1.6 3.9-1.6 6 0s3.9 1.6 6 0 3-1.2 3-1.2" fill="none" stroke="#74cfe6" strokeWidth="1.6" strokeLinecap="round" />
          <path d="M11 3.5v4.2" stroke="#f2b04c" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
        <span>CoastWatch</span>
      </a>
      <nav className="mh-nav" aria-label="Primary">
        {["Map", "My Coast", "Blooms", "Fisheries"].map((n) => (
          <a key={n} href={n === "Map" ? "#/explorer" : n === "Blooms" ? "#/station" : "#/"} aria-current={n === "Map" ? "page" : undefined}>{n}</a>
        ))}
      </nav>
      <div className="mh-right">
        <div className="mh-mode" role="radiogroup" aria-label="Presentation">
          {(["coastal", "research"] as const).map((m) => (
            <button key={m} role="radio" aria-checked={mode === m} onClick={() => onMode(m)}
              onKeyDown={(e) => { if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); onMode(m === "coastal" ? "research" : "coastal"); } }}
              tabIndex={mode === m ? 0 : -1}>
              {m === "coastal" ? "Coastal" : "Research"}
            </button>
          ))}
        </div>
        <button className="mh-notices" onClick={onNotices} aria-haspopup="dialog">
          <span className="mark mark-rule" aria-hidden="true" />
          <span className="num">{notices ?? "…"}</span>
          <span className="mh-notices-l">official notices</span>
          {unverified && <span className="mh-unv">not verified</span>}
        </button>
        <button className="mh-menu btn-quiet" aria-expanded={menu} onClick={() => setMenu(!menu)} aria-label="More">⋯</button>
      </div>
      <p className="mh-status">
        {compact ? "Notices may be incomplete. Before harvesting, check" : "Notices are copied from CDFW and CDPH and may be incomplete. Before harvesting or eating seafood, check"}{" "}
        <a href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" target="_blank" rel="noopener" className="ext">CDFW</a> and{" "}
        <a href="https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" target="_blank" rel="noopener" className="ext">CDPH</a>
        <span className="mh-status-phone"> · Shellfish line <a href="tel:+18005534133">(800) 553-4133</a></span>
      </p>
    </header>
  );
}
