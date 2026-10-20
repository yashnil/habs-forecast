import type { NextConfig } from "next";

// names this build for the service worker (/sw.js?b=…), whose code cache is one per build
const BUILD = (process.env.VERCEL_GIT_COMMIT_SHA ?? process.env.NERITIC_BUILD ?? Date.now().toString(36)).slice(0, 12);

const nextConfig: NextConfig = {
  transpilePackages: ["react-map-gl", "mapbox-gl"],
  env: { NEXT_PUBLIC_NERITIC_BUILD: BUILD },
  async headers() {
    // the icons never change under the same name between deploys of the same brand: no revalidation per visit
    const year = "public, max-age=31536000, immutable";
    return ["/favicon.ico", "/icon.svg", "/apple-icon.png"].map((source) => ({ source, headers: [{ key: "Cache-Control", value: year }] }));
  },
};

export default nextConfig;
