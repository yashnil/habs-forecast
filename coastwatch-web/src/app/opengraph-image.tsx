import { ImageResponse } from "next/og";
import { BRAND } from "@/content/brand";

// The sharing card: the mark, the name and the tagline on the masthead's navy, with the
// research-preview status said plainly. Rendered at build time with next/og's bundled font.
export const alt = `${BRAND.name}: ${BRAND.tagline}`;
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "space-between", background: "#06111e", color: "#ffffff", padding: "72px 80px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 28 }}>
          <svg width="112" height="112" viewBox="0 0 32 32">
            <rect width="32" height="32" rx="8" fill="#10263e" />
            <defs>
              <linearGradient id="lit" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor="#6cc6dc" stopOpacity=".42" />
                <stop offset="1" stopColor="#6cc6dc" stopOpacity=".04" />
              </linearGradient>
            </defs>
            <path d="M6 10H19.5V21H6Z" fill="url(#lit)" />
            <path d="M6 10H26" stroke="#6cc6dc" strokeWidth="2" strokeLinecap="round" />
            <path d="M6 21H19.5L24.5 26.5" fill="none" stroke="#e8eef5" strokeOpacity=".78" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="12.75" cy="15.5" r="2.2" fill="#f6bb5c" />
          </svg>
          <div style={{ display: "flex", fontSize: 88, fontWeight: 700, letterSpacing: -1.5 }}>Neritic</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div style={{ fontSize: 44, color: "#e8eef5" }}>{BRAND.tagline}</div>
          <div style={{ fontSize: 26, color: "#74cfe6" }}>California coast · harmful algal blooms, fisheries and the water · research preview</div>
        </div>
        <div style={{ display: "flex", fontSize: 21, color: "#8193ac" }}>
          Independent research project · not affiliated with NOAA, CDFW or CDPH
        </div>
      </div>
    ),
    size,
  );
}
