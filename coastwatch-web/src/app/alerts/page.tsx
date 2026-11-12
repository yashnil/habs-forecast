import { redirect } from "next/navigation";

/**
 * `/alerts`: where the bar's Alerts link goes (src/nx/shell/unread.ts). Until the alerts page is built here,
 * it opens the alerts section of Your data, carrying the port, so the link is never a dead end.
 */
export default async function AlertsPage({ searchParams }: { searchParams: Promise<{ place?: string }> }) {
  const { place } = await searchParams;
  redirect(place && /^port-\d{1,4}$/.test(place) ? `/account?place=${place}#alerts` : "/account#alerts");
}
