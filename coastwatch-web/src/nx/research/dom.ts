"use client";

import type { Strip } from "./model";
import { layoutStrips } from "./RecordStrips";

/** Save a file the page made (a CSV, a figure) through the browser. */
export function download(name: string, text: string, type: string) {
  const blob = new Blob([text], { type: `${type};charset=utf-8` });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/** The presentation properties a standalone figure needs, read from the live page (its mode's tokens resolved). */
const PROPS = ["fill", "stroke", "stroke-width", "stroke-dasharray", "opacity", "font-family", "font-size", "font-weight", "text-anchor", "dominant-baseline", "shape-rendering"] as const;

/**
 * The strips as one standalone SVG figure: the live drawing with its colours and type inlined (so it reads
 * the same outside the page), each strip's name and unit written above it, a title and the citation below.
 */
export function svgFigure(svg: SVGSVGElement, o: { title: string; strips: Strip[]; plotH: number; folded: ReadonlySet<string>; caption: string }): string {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const live = svg.querySelectorAll<SVGElement>("*");
  const copies = clone.querySelectorAll<SVGElement>("*");
  live.forEach((el, i) => {
    const cs = getComputedStyle(el);
    const c = copies[i];
    const style = PROPS.map((p) => {
      const v = cs.getPropertyValue(p);
      return v ? `${p}:${v}` : "";
    })
      .filter(Boolean)
      .join(";");
    if (style) c.setAttribute("style", style);
    c.removeAttribute("class");
  });
  // the clip that drew the record in is not needed in a still
  clone.querySelectorAll("[clip-path]").forEach((g) => g.removeAttribute("clip-path"));
  clone.querySelector("defs")?.remove();
  const w = Number(svg.getAttribute("width"));
  const h = Number(svg.getAttribute("height"));
  const top = 40;
  const ink = getComputedStyle(svg).getPropertyValue("color") || "currentColor";
  const font = getComputedStyle(svg).getPropertyValue("font-family");
  const paper = getComputedStyle(document.querySelector("[data-chart-mode]") ?? document.body).getPropertyValue("--paper").trim() || "none";
  const esc = (t: string) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const heads = layoutStrips(o.strips, o.plotH, o.folded).rows.map((r) => `<text x="0" y="${top + r.top + 20}" style="font-family:${font};font-size:15px;font-weight:600;fill:${ink}">${esc(`${r.strip.kind === "model" ? "Model · " : ""}${r.strip.label}`)} <tspan style="font-weight:400">${esc(r.folded ? "no sample in this window" : r.strip.unit)}</tspan></text>`);
  const lines = wrap(o.caption, Math.max(40, Math.floor(w / 7.4)));
  const foot = lines.map((l, i) => `<text x="0" y="${top + h + 22 + i * 18}" style="font-family:${font};font-size:14px;fill:${ink}">${esc(l)}</text>`);
  const H = top + h + 22 + lines.length * 18 + 8;
  clone.removeAttribute("aria-hidden");
  clone.setAttribute("x", "0");
  clone.setAttribute("y", String(top));
  return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${H}" viewBox="0 0 ${w} ${H}"><rect width="100%" height="100%" style="fill:${paper}"/><text x="0" y="22" style="font-family:${font};font-size:18px;font-weight:600;fill:${ink}">${esc(o.title)}</text>${heads.join("")}${new XMLSerializer().serializeToString(clone)}${foot.join("")}</svg>\n`;
}

function wrap(text: string, n: number): string[] {
  const out: string[] = [];
  let cur = "";
  for (const word of text.split(/\s+/)) {
    if ((cur + " " + word).trim().length > n && cur) {
      out.push(cur);
      cur = word;
    } else cur = (cur + " " + word).trim();
  }
  if (cur) out.push(cur);
  return out;
}
