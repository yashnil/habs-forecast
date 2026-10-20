import type { NextConfig } from "next";

// names this build for the service worker (/sw.js?b=…), whose code cache is one per build
const BUILD = (process.env.VERCEL_GIT_COMMIT_SHA ?? process.env.NERITIC_BUILD ?? Date.now().toString(36)).slice(0, 12);

const nextConfig: NextConfig = {
  transpilePackages: ["react-map-gl", "mapbox-gl"],
  env: { NEXT_PUBLIC_NERITIC_BUILD: BUILD },
  async headers() {
    // the icons never change under the same name between deploys of the same brand: no revalidation per visit
    // a day, not a year: the mark is provisional and a new one must reach returning browsers
    const day = "public, max-age=86400";
    return ["/favicon.ico", "/icon.svg", "/apple-icon.png"].map((source) => ({ source, headers: [{ key: "Cache-Control", value: day }] }));
  },
};

export default nextConfig;
