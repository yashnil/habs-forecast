import { ImageResponse } from "next/og";
import { BRAND } from "@/content/brand";
import { C2, C2_TONES } from "@/v4/brand/c2";

// The sharing card: the C2 mark and drawn wordmark, the tagline on the masthead's navy, with the
// research-preview status said plainly. Rendered at build time with next/og's bundled font.
export const alt = `${BRAND.name}: ${BRAND.tagline}`;
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "space-between", background: "#06111e", color: "#ffffff", padding: "72px 80px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 22 }}>
          <svg width="112" height="112" viewBox="0 0 48 48">
            <path d={C2.tile} fill={C2_TONES.dark.tile} />
            <path d={C2.stubs[0]} stroke={C2_TONES.dark.stub} strokeWidth={C2.stubWidth} />
            <path d={C2.stubs[1]} stroke={C2_TONES.dark.stub} strokeWidth={C2.stubWidth} />
            <path transform={C2.nAt} d={C2.n} fill={C2_TONES.dark.n} />
          </svg>
          {/* the drawn wordmark, as in the header (outlined; no font needed) */}
          <svg width={(C2.lockupWidth - 58) * (112 / 48)} height="112" viewBox={`58 0 ${C2.lockupWidth - 58} 48`}>
            <path transform={C2.wordAt} d={C2.word} fill={C2_TONES.dark.word} />
          </svg>
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
