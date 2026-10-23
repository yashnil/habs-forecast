import { ImageResponse } from "next/og";
import { ShareCard, shareCardAlt, shareCardSize } from "@/v4/brand/ShareCard";

// The sharing card (src/v4/brand/ShareCard.tsx), rendered at build time.
export const alt = shareCardAlt;
export const size = shareCardSize;
export const contentType = "image/png";

export default function Image() {
  return new ImageResponse(<ShareCard />, size);
}
