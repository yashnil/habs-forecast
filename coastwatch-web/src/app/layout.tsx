import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Newsreader } from "next/font/google";
import "./globals.css";
import { BRAND } from "@/content/brand";

// Self-hosted by next/font at build time: no runtime request to Google.
const plexSans = IBM_Plex_Sans({ variable: "--font-plex-sans", subsets: ["latin"], weight: ["400", "500", "600"] });
const plexMono = IBM_Plex_Mono({ variable: "--font-plex-mono", subsets: ["latin"], weight: ["400", "500"] });
// variable, with optical sizing: sharper display sizes, steadier small headlines (signature §typography)
const newsreader = Newsreader({ variable: "--font-newsreader", subsets: ["latin"], axes: ["opsz"], style: ["normal"] });

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
    <html lang="en">
      <body className={`${plexSans.variable} ${plexMono.variable} ${newsreader.variable} antialiased`}>{children}</body>
    </html>
  );
}
