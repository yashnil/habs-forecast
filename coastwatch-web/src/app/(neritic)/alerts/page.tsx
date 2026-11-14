import { AlertsPage } from "@/nx/alerts/AlertsPage";

export const metadata = { title: "Alerts", robots: { index: false } };

/**
 * `/alerts?place=port-N`: the ports the reader follows and the alerts this device has made (src/nx/alerts).
 * Everything it shows is in the reader's browser, so the server renders only the frame; `place` (the bar's
 * Alerts link carries the current port) preselects that port to follow.
 */
export default function Alerts() {
  return <AlertsPage />;
}
