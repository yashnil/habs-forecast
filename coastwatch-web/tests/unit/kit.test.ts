import { describe, expect, it } from "vitest";
import { createElement as h } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Tag } from "@/v4/ui/Tag";
import { Figure } from "@/v4/ui/Figure";
import { Freshness } from "@/v4/ui/Freshness";
import { Unverified } from "@/v4/ui/Unverified";
import { Button, LinkButton } from "@/v4/ui/Button";
import { SectionHead } from "@/v4/ui/SectionHead";

/** The v11 kit (src/v4/ui): the words carry the meaning, never colour alone. */
describe("kit: Tag", () => {
  it("each level is a word and a shape on a data-level tint", () => {
    const html = renderToStaticMarkup(h(Tag, { level: "delayed" }, "Season delayed"));
    expect(html).toContain('data-level="delayed"');
    expect(html).toContain("Season delayed");
    expect(html).toContain("<svg");
  });
  it('the calmest level defaults to "None on record"', () => {
    expect(renderToStaticMarkup(h(Tag, { level: "none" }))).toContain("None on record");
  });
});

describe("kit: Unverified", () => {
  it("says the words, with the count when given", () => {
    expect(renderToStaticMarkup(h(Unverified))).toContain("Not verified");
    expect(renderToStaticMarkup(h(Unverified, { count: "7 of 7" }))).toContain("7 of 7 not verified");
  });
  it("its explanation stays in the DOM and links the agencies", () => {
    const html = renderToStaticMarkup(h(Unverified));
    expect(html).toContain("popover");
    expect(html).toContain("wildlife.ca.gov");
    expect(html).toContain("cdph.ca.gov");
  });
});

describe("kit: Figure", () => {
  it("a stale value says (old) on the value, in words", () => {
    expect(renderToStaticMarkup(h(Figure, { value: "5.5", unit: "ppm", stale: true, testId: "f" }))).toContain("(old)");
    expect(renderToStaticMarkup(h(Figure, { value: "5.5", unit: "ppm" }))).not.toContain("(old)");
  });
});

describe("kit: Freshness", () => {
  const items = [
    { label: "Tests", when: "Sep 28", stale: true },
    { label: "NWS", when: "2:34 AM" },
  ];
  it("names only what is old on the cue; every source is in the panel", () => {
    const html = renderToStaticMarkup(h(Freshness, { items }));
    const cue = html.slice(0, html.indexOf('popover="auto"'));
    expect(cue).toContain("Tests Sep 28 (old)");
    expect(cue).not.toContain("NWS");
    expect(html).toContain("NWS");
  });
  it("with nothing old it is one quiet word", () => {
    const html = renderToStaticMarkup(h(Freshness, { items: [{ label: "NWS", when: "2:34 AM" }] }));
    expect(html.slice(0, html.indexOf('popover="auto"'))).toContain("Sources");
  });
});

describe("kit: Button, LinkButton, SectionHead", () => {
  it("buttons carry their kind and an external link opens a new tab", () => {
    expect(renderToStaticMarkup(h(Button, { kind: "primary", children: "Go" }))).toContain("nx-btn-primary");
    const a = renderToStaticMarkup(h(LinkButton, { kind: "tertiary", href: "https://example.gov", external: true, children: "Source" }));
    expect(a).toContain('target="_blank"');
    expect(a).toContain("nx-btn-tertiary");
  });
  it("a section head is a sentence-case serif title", () => {
    const head = renderToStaticMarkup(h(SectionHead, { title: "Sea", as: "h3", id: "sea-h" }));
    expect(head).toContain('<h3 id="sea-h" class="nx-title">Sea</h3>');
  });
});
