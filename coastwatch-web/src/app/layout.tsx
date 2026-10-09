import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Newsreader } from "next/font/google";
import "./globals.css";

// Self-hosted by next/font at build time: no runtime request to Google.
const plexSans = IBM_Plex_Sans({ variable: "--font-plex-sans", subsets: ["latin"], weight: ["400", "500", "600"] });
const plexMono = IBM_Plex_Mono({ variable: "--font-plex-mono", subsets: ["latin"], weight: ["400", "500"] });
const newsreader = Newsreader({ variable: "--font-newsreader", subsets: ["latin"], weight: ["400", "500"], style: ["normal"] });

export const metadata: Metadata = {
  title: "CoastWatch — California coastal bloom forecasts",
  description:
    "Official C-HARM harmful algal bloom and domoic acid forecast probabilities for the California coast, with satellite chlorophyll and landing ports. Not an official source; closures and advisories come from CDFW and CDPH.",
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
