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
            <path d="M5 10.5c2.6-1.7 5.2-1.7 7.8 0s5.4 1.7 8.2 0 4.6-1.6 6 -.6" fill="none" stroke="#6cc6dc" strokeWidth="2" strokeLinecap="round" />
            <path d="M4.5 23.5h11.2c1.6 0 2.6.5 3.5 1.6l2.6 3.2" fill="none" stroke="#e8eef5" strokeOpacity=".6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="10.4" cy="18" r="2" fill="#f6bb5c" />
            <path d="M14 15.6a4 4 0 0 1 0 4.8" fill="none" stroke="#f6bb5c" strokeOpacity=".75" strokeWidth="1.6" strokeLinecap="round" />
            <path d="M16.7 14.1a7 7 0 0 1 0 7.8" fill="none" stroke="#f6bb5c" strokeOpacity=".4" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
          <div style={{ display: "flex", fontSize: 76, letterSpacing: -1.5 }}>
            <span style={{ fontWeight: 700 }}>Neritic</span>
            <span style={{ marginLeft: 20, opacity: 0.9 }}>Signal</span>
          </div>
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
