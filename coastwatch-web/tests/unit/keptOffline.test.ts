import { describe, expect, it } from "vitest";
import { staticRefs } from "@/v4/offline/kept";

describe("kept briefing offline: the files /coast/kept loads", () => {
  it("finds scripts, styles and fonts, with and without the leading /_next/", () => {
    const html = `<link rel="stylesheet" href="/_next/static/css/a1b2.css"><script src="/_next/static/chunks/app/coast/kept/page-9f.js" async></script>
      <script>self.__next_f.push([1,"2:I[\\"static/chunks/1234-ab.js\\",\\"static/chunks/app/layout-cd.js\\"]"])</script>`;
    const refs = staticRefs(html);
    expect(refs).toContain("/_next/static/css/a1b2.css");
    expect(refs).toContain("/_next/static/chunks/app/coast/kept/page-9f.js");
    expect(refs).toContain("/_next/static/chunks/1234-ab.js");
    expect(refs).toContain("/_next/static/chunks/app/layout-cd.js");
  });
  it("reads font files out of a stylesheet and nothing from another origin", () => {
    const css = `@font-face{src:url(/_next/static/media/plex-s.p.woff2) format("woff2")} .x{background:url(https://example.com/static/chunks/evil.js)}`;
    expect(staticRefs(css)).toEqual(["/_next/static/media/plex-s.p.woff2"]);
  });
});
