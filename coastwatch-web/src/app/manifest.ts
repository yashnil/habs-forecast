import type { MetadataRoute } from "next";
import { BRAND } from "@/content/brand";

/** Web app manifest: the brand's name and icons (scripts/build_brand_assets.mjs). */
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: `${BRAND.name} — research preview`,
    short_name: BRAND.name,
    description: BRAND.tagline,
    start_url: "/",
    display: "standalone",
    background_color: "#f5f3ee",
    theme_color: "#06111e",
    icons: [
      { src: "/icon.svg", type: "image/svg+xml", sizes: "any" },
      { src: "/icons/icon-192.png", type: "image/png", sizes: "192x192" },
      { src: "/icons/icon-512.png", type: "image/png", sizes: "512x512" },
      { src: "/icons/maskable-512.png", type: "image/png", sizes: "512x512", purpose: "maskable" },
    ],
  };
}
