import type { NextConfig } from "next";

// Portfolio-preview builds (NEXT_PUBLIC_CW_DEMO=1) show the map only; see src/lib/demo.ts.
const demo = process.env.NEXT_PUBLIC_CW_DEMO === "1";

const nextConfig: NextConfig = {
  transpilePackages: ["react-map-gl", "mapbox-gl"],
  async redirects() {
    return demo
      ? ["/bloom", "/fisheries"].map((source) => ({ source, destination: "/", permanent: false }))
      : [];
  },
};

export default nextConfig;
