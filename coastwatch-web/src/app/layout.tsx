import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Newsreader } from "next/font/google";
import "./globals.css";
import { BRAND } from "@/content/brand";
import { DAYLIGHT_SCRIPT } from "@/v4/shell/daylight";
import { pilotFrontDoor } from "@/pilot/frontDoor";
import { LegacyChrome } from "@/nx/shell/LegacyChrome";

// Self-hosted by next/font at build time: no runtime request to Google.
// v7: the v2 faces (IBM Plex Sans, Newsreader) are no longer loaded; theme.css maps their variables
// to the v4 faces, so v2 components still resolve. Plex Mono stays for tables and coordinates.
const plexMono = IBM_Plex_Mono({ variable: "--font-plex-mono", subsets: ["latin"], weight: ["400", "500"] });
// v11 Open Water (src/v4/theme.css): the interface face and the editorial serif for places, titles and
// margin notes. All SIL OFL 1.1.
const plexSans = IBM_Plex_Sans({ variable: "--font-plex-sans-v10", subsets: ["latin"], weight: ["300", "400", "500", "600"] });
const newsreader = Newsreader({ variable: "--font-newsreader-v10", subsets: ["latin"], axes: ["opsz"], weight: "variable", style: ["normal", "italic"] });

const DESCRIPTION =
  "Coastal intelligence, grounded in evidence. A research preview for the California coast: NOAA's C-HARM bloom and domoic acid probabilities, shore measurements and satellite chlorophyll, with links to official CDFW and CDPH notices. Not an official source and not a seafood-safety service; closures and advisories come from CDFW and CDPH. An independent research project, not affiliated with NOAA, CDFW or CDPH.";

export const metadata: Metadata = {
  metadataBase: new URL(process.env.CW_SITE_URL ?? "https://coastwatch-demo.vercel.app"),
  applicationName: BRAND.name,
  title: { default: `${BRAND.name} (research preview) — coastal intelligence for California`, template: `%s — ${BRAND.name}` },
  description: DESCRIPTION,
  openGraph: { type: "website", siteName: BRAND.name, title: `${BRAND.name} — ${BRAND.tagline}`, description: DESCRIPTION, locale: "en_US" },
  twitter: { card: "summary_large_image", title: `${BRAND.name} — ${BRAND.tagline}`, description: DESCRIPTION },
};

export const viewport: Viewport = {
  themeColor: "#071722",
  colorScheme: "dark",
};
// (the route group sets its own: the chart's paper, and loads its own two faces, src/nx/ds/fonts.ts)

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    // the font variables live on <html>, so the v4 tokens defined at :root can resolve them;
    // suppressHydrationWarning: the launch screen's and daylight mode's inline scripts may mark <html> before hydration
    <html lang="en" suppressHydrationWarning className={`${plexMono.variable} ${plexSans.variable} ${newsreader.variable}`}>
      <body className="antialiased">
        {/* daylight mode before the first paint (src/v4/shell/daylight.ts) */}
        <script dangerouslySetInnerHTML={{ __html: DAYLIGHT_SCRIPT }} />
        {/* v12: the route group (/, Today, Explore, My Coast, Research) has its own shell and one entry; the old
            launch film is gone, and the legacy pages keep the v4 root (src/nx/shell/LegacyChrome.tsx) */}
        <LegacyChrome pilot={pilotFrontDoor()}>{children}</LegacyChrome>
      </body>
    </html>
  );
}
