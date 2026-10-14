import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Newsreader } from "next/font/google";
import "./globals.css";

// Self-hosted by next/font at build time: no runtime request to Google.
const plexSans = IBM_Plex_Sans({ variable: "--font-plex-sans", subsets: ["latin"], weight: ["400", "500", "600"] });
const plexMono = IBM_Plex_Mono({ variable: "--font-plex-mono", subsets: ["latin"], weight: ["400", "500"] });
// variable, with optical sizing: sharper display sizes, steadier small headlines (signature §typography)
const newsreader = Newsreader({ variable: "--font-newsreader", subsets: ["latin"], axes: ["opsz"], style: ["normal"] });

export const metadata: Metadata = {
  title: "CoastWatch (research preview) — harmful algal blooms on the California coast",
  description:
    "A research preview: NOAA's C-HARM bloom and domoic acid probabilities, shore measurements and satellite chlorophyll for the California coast, with links to official CDFW and CDPH notices. Not an official source and not a seafood-safety service; closures and advisories come from CDFW and CDPH.",
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
