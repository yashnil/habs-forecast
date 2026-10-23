import type { Metadata, Viewport } from "next";
import { Atkinson_Hyperlegible_Next, IBM_Plex_Mono, Source_Serif_4 } from "next/font/google";
import "./globals.css";
import { BRAND } from "@/content/brand";
import { SplashDecide } from "@/v4/shell/Splash";
import { MotionRoot } from "@/v4/shell/MotionRoot";
import { DAYLIGHT_SCRIPT } from "@/v4/shell/daylight";

// Self-hosted by next/font at build time: no runtime request to Google.
// v7: the v2 faces (IBM Plex Sans, Newsreader) are no longer loaded; theme.css maps their variables
// to the v4 faces, so v2 components still resolve. Plex Mono stays for tables and coordinates.
const plexMono = IBM_Plex_Mono({ variable: "--font-plex-mono", subsets: ["latin"], weight: ["400", "500"] });
// v4: the interface and reading face, the editorial serif (src/v4/theme.css)
const atkinson = Atkinson_Hyperlegible_Next({ variable: "--font-atkinson", subsets: ["latin"], weight: "variable" });
const sourceSerif = Source_Serif_4({ variable: "--font-source-serif", subsets: ["latin"], axes: ["opsz"], weight: "variable" });

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
  themeColor: "#06111e",
  colorScheme: "light",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    // the font variables live on <html>, so the v4 tokens defined at :root can resolve them;
    // suppressHydrationWarning: the launch screen's and daylight mode's inline scripts may mark <html> before hydration
    <html lang="en" suppressHydrationWarning className={`${plexMono.variable} ${atkinson.variable} ${sourceSerif.variable}`}>
      <body className="antialiased">
        {/* daylight mode before the first paint (src/v4/shell/daylight.ts) */}
        <script dangerouslySetInnerHTML={{ __html: DAYLIGHT_SCRIPT }} />
        <SplashDecide />
        <MotionRoot>{children}</MotionRoot>
      </body>
    </html>
  );
}
