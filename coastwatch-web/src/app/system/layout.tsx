import type { Metadata } from "next";
import type { ReactNode } from "react";
import "@/nx/ds/tokens.css";

// The living catalog of the v12 design system: a review and screenshot surface, never indexed.
export const metadata: Metadata = {
  title: "Component catalog",
  robots: { index: false, follow: false },
};

export default function SystemLayout({ children }: { children: ReactNode }) {
  // the two faces' variables are on <html> (src/app/layout.tsx)
  return children;
}
