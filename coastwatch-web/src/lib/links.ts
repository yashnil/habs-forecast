/**
 * External links from published data (provenance, notice sources) are rendered as anchors.
 * Only http(s), mailto and tel are allowed, so a malformed or hostile URL in a dataset can
 * never become a `javascript:` link. Anything else renders as an inert "#".
 */
export function safeHref(url: string | null | undefined): string {
  if (!url) return "#";
  try {
    const u = new URL(url);
    return ["https:", "http:", "mailto:", "tel:"].includes(u.protocol) ? u.href : "#";
  } catch {
    return "#";
  }
}
