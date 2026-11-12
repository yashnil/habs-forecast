/** Where the bar's Alerts link goes, carrying the port. The unread count beside it is the alerts feature's own (`useUnread` in src/nx/alerts/useInbox.ts). */
export const alertsHref = (key: string | null) => (key ? `/alerts?place=${encodeURIComponent(key)}` : "/alerts");
