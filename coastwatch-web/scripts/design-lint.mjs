#!/usr/bin/env node
/**
 * npm run lint:design — the anti-vibe-coding rules of the Neritic design system, enforced mechanically
 * (docs/coastwatch/v12/design-system/README.md section 3, as amended for the v14 synthesis in
 * docs/coastwatch/v14/design-system/README.md section 8: Figtree at 11 px and up, objects float with the token
 * shadows and radii, the map voice's spaced capitals live in the design system and the chart). Scans src/nx/**, src/app/**\/(neritic)/** and the
 * /system catalog. CSS is read declaration by declaration; TS/TSX through the TypeScript AST, so only real
 * string literals, template text and JSX text count (comments never do).
 *
 * A line may be excused with a reason, on the line itself or the line above:
 *   /* design-lint-allow <rule>: <why> *\/     or     // design-lint-allow <rule>: <why>
 * An allowance without a reason is itself an error.
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

export const RULES = {
  "colour-literal": "Colour literal outside the token files; use a token from src/nx/ds/tokens.css",
  "font-size": "Text under 11 px; the synthesis scale is 11 / 13 / 15 / 18 / 25 / 40 / 54 / 68 (--fs-*)",
  decoration: "Backdrop filters and raw gradients are not part of the system; shadows and hatches come from tokens (--shadow*, --hatch-*)",
  radius: "Radius must be a token (--r-card 20, --r-ctl 12, --r-sm 9, --r-bar, --r-pill, --r-tag, --r-sheet, --r-0, --r-1), 0 or 50%",
  caps: "Capitals and spaced labels are the design system's (ds Micro, the chart's map voice); a page writes sentence case",
  emoji: "No emoji",
  "forbidden-word": "Forbidden word (truth rule T36); Neritic never gives a verdict",
  "iso-date": "Hard-coded ISO date in a component; dates come from data and are formatted by ds/format",
  motion: "Motion goes through src/nx/ds/motion presets: no raw transition over 200 ms, no keyframes outside ds/",
  "motion-preset": "Inline Motion timing; use a preset from src/nx/ds/motion.ts (ink, pan, settle, register, sheet, press, cut)",
  "font-family": "Type must use --font (Figtree; --font-sans and --font-serif are its aliases); no monospace, no other families",
  "space-scale": "Spacing must be on the 2 px grid up to 64 px, or a --s-* token",
  "allow-reason": "design-lint-allow needs a rule and a reason",
  // v13 (docs/coastwatch/v13/design-system/README.md section 10)
  "fixed-position": "position: fixed belongs to the shell and the design system; a page lays out in its Desk, margin or a ChartDock",
  "margin-width": "--margin-w is the workspace's; a page never sizes itself by the margin column",
  "raw-chart": "Charts are drawn with the plot kit (src/nx/ds/plot); d3 is imported only in src/nx/ds and src/nx/chart",
  "one-primary": "One primary button per view; make the others secondary or quiet",
  "raw-button": "A hand-styled <button>; use Button, IconButton, Segmented, Disclosure or OptionList from src/nx/ds",
  "small-text": "Many uses of 14 px text in one stylesheet; 14 px is for provenance, axis ticks and footnotes (--type-small)",
};

/** Reported and counted, but they do not fail the run: pages are moving onto the v13 parts (README section 10 says when each becomes an error). */
export const WARN_RULES = new Set(["fixed-position", "raw-chart", "one-primary", "raw-button", "small-text"]);
/** Where the shell's and the design system's own layout and controls live: the page rules do not apply there. */
const SYSTEM_DIR = /src\/nx\/(ds|shell|chart)\/|src\/app\/system\//;
const SMALL_TEXT_BUDGET = 6;

const TOKEN_FILES = [/src\/nx\/ds\/tokens\.css$/, /src\/nx\/ds\/palette\.ts$/];
const MOTION_HOME = /src\/nx\/ds\/motion\.ts$/;
const DS_DIR = /src\/nx\/ds\//;

const FORBIDDEN = [
  "all clear",
  "no-go",
  "go/no-go",
  "good day",
  "good to fish",
  "best spot",
  "best place to fish",
  "hot ?spot",
  "recommended",
  "expected catch",
  "price forecast",
  "green light",
  "low risk",
  "bloom risk",
  "toxicity",
  "toxin forecast",
  "ai-powered",
  "accurate",
  "(?:un)?safe",
  "go",
];
const FORBIDDEN_RE = new RegExp(`(?<![\\p{L}\\p{N}_-])(${FORBIDDEN.join("|")})(?![\\p{L}\\p{N}_-])`, "iu");
const HEX_RE = /(^|[^\w&/])#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})(?![\w-])/;
const FN_COLOUR_RE = /\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(/i;
const NAMED_COLOUR_RE = /(^|[\s:,(])(white|black|red|green|blue|gray|grey|yellow|orange|purple|pink|silver|navy|teal)(?=$|[\s;,)!])/i;
const GRADIENT_RE = /\b(?:repeating-)?(?:linear|radial|conic)-gradient\(/i;
const EMOJI_RE = /\p{Emoji_Presentation}|\p{Extended_Pictographic}️/u;
const ISO_RE = /\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b/;
const onGrid = (v) => Math.abs(v) <= 64 && Math.abs(v) % 2 === 0;
const FONT_VAR = /var\(--font(-serif|-sans)?\)/;
const RADIUS_OK = (v) => v === "0" || v === "50%" || /^var\(--r-[a-z0-9-]+\)$/.test(v) || /^calc\(.*var\(--r-/.test(v);

const lineOf = (text, offset) => text.slice(0, offset).split("\n").length;

/** Allowances: line number -> set of rules excused on that line and the next. */
function allowances(text, file, out) {
  const map = new Map();
  const re = /design-lint-allow\s+([a-z-]+)?\s*(?::\s*(\S[^*\n]*))?/g;
  let m;
  while ((m = re.exec(text))) {
    const line = lineOf(text, m.index);
    if (!m[1] || !RULES[m[1]] || !m[2] || m[2].trim().length < 4) {
      out.push({ file, line, rule: "allow-reason", message: RULES["allow-reason"] });
      continue;
    }
    for (const l of [line, line + 1]) {
      if (!map.has(l)) map.set(l, new Set());
      map.get(l).add(m[1]);
    }
  }
  return map;
}

function durationsMs(value) {
  const out = [];
  const re = /(\d*\.?\d+)(ms|s)\b/g;
  let m;
  while ((m = re.exec(value))) out.push(m[2] === "s" ? parseFloat(m[1]) * 1000 : parseFloat(m[1]));
  return out;
}

function pxValue(v) {
  const m = /^(-?\d*\.?\d+)(px|rem)$/.exec(v.trim());
  if (!m) return null;
  return m[2] === "rem" ? parseFloat(m[1]) * 16 : parseFloat(m[1]);
}

/** Colour, emoji, forbidden-word, ISO and decoration checks on a piece of user-facing or style text. */
function textChecks(s, { copy, style, iso }) {
  const hits = [];
  if (HEX_RE.test(s) || FN_COLOUR_RE.test(s)) hits.push("colour-literal");
  if (GRADIENT_RE.test(s) || (style && /\bbox-shadow\s*:|backdrop-filter/.test(s))) hits.push("decoration");
  if (EMOJI_RE.test(s)) hits.push("emoji");
  if (copy && FORBIDDEN_RE.test(s)) hits.push("forbidden-word");
  if (iso && ISO_RE.test(s)) hits.push("iso-date");
  return hits;
}

/* ---------------- CSS ---------------- */

function lintCss(file, text, tokenFile, inDs) {
  const out = [];
  const system = SYSTEM_DIR.test(file);
  const small = [];
  const clean = text.replace(/\/\*[\s\S]*?\*\//g, (c) => c.replace(/[^\n]/g, " "));
  if (!inDs) {
    const kf = /@(?:-webkit-)?keyframes\b/g;
    let m;
    while ((m = kf.exec(clean))) out.push({ file, line: lineOf(clean, m.index), rule: "motion", message: `${RULES.motion} (@keyframes)` });
  }
  const block = /\{([^{}]*)\}/g;
  let b;
  while ((b = block.exec(clean))) {
    const body = b[1];
    const base = b.index + 1;
    const decl = /([a-zA-Z-]+)\s*:\s*([^;]+)/g;
    let d;
    while ((d = decl.exec(body))) {
      const prop = d[1].toLowerCase();
      const value = d[2].trim();
      const line = lineOf(clean, base + d.index);
      const add = (rule, extra = "") => out.push({ file, line, rule, message: `${RULES[rule]}${extra ? ` (${extra})` : ""}`, text: `${prop}: ${value}` });
      if (prop.startsWith("--")) {
        if (!tokenFile && (HEX_RE.test(` ${value}`) || FN_COLOUR_RE.test(value))) add("colour-literal", prop);
        continue;
      }
      if (!tokenFile && (HEX_RE.test(` ${value}`) || FN_COLOUR_RE.test(value) || (/(color|background|fill|stroke|border|outline|caret|shadow)/.test(prop) && NAMED_COLOUR_RE.test(` ${value}`)))) add("colour-literal", `${prop}: ${value}`);
      if (prop === "font-size") {
        const px = pxValue(value);
        if (px !== null && px < 11) add("font-size", value);
      }
      if (prop === "font") {
        const m = /(\d*\.?\d+)(px|rem)\b/.exec(value);
        if (m) {
          const px = m[2] === "rem" ? parseFloat(m[1]) * 16 : parseFloat(m[1]);
          if (px < 11) add("font-size", value);
        }
        if (!FONT_VAR.test(value) && value !== "inherit") add("font-family", value);
      }
      if (prop === "font-family" && !new RegExp(`^${FONT_VAR.source}$`).test(value) && value !== "inherit") add("font-family", value);
      if (/monospace/i.test(value)) add("font-family", value);
      if (prop.endsWith("backdrop-filter")) add("decoration", prop);
      if (GRADIENT_RE.test(value)) add("decoration", "gradient");
      // the design system and the chart draw their own small shapes (a glyph's 3 px corners, a chip's 13 px);
      // a page uses the radius tokens
      if (/^border(-[a-z]+)*-radius$/.test(prop) && !system) {
        const ok = value.split(/\s+/).every(RADIUS_OK);
        if (!ok) add("radius", value);
      }
      if (!system && prop === "text-transform" && /uppercase/.test(value)) add("caps", value);
      if (/^font-variant(-caps)?$/.test(prop) && /small-caps|all-small-caps|petite-caps|unicase|titling-caps/.test(value)) add("caps", value);
      if (prop === "letter-spacing" && !system) {
        const m = /^(-?\d*\.?\d+)(em|px)$/.exec(value);
        if (m && ((m[2] === "em" && parseFloat(m[1]) > 0.04) || (m[2] === "px" && parseFloat(m[1]) > 0.6))) add("caps", `letter-spacing ${value}`);
      }
      if (/^(transition|transition-duration|animation|animation-duration)$/.test(prop)) {
        if (durationsMs(value).some((x) => x > 200)) add("motion", value);
      }
      if (!system && prop === "position" && /^fixed\b/.test(value)) add("fixed-position");
      if (!system && /var\(--margin-w\b/.test(value)) add("margin-width", prop);
      if (!system && /var\(--(t-14|type-small)\)/.test(value) && /^font(-size)?$/.test(prop)) small.push(line);
      if (!system && /^(margin|padding|gap|row-gap|column-gap)(-(top|right|bottom|left|inline|block)(-start|-end)?)?$/.test(prop)) {
        const bad = value.split(/\s+/).map(pxValue).filter((v) => v !== null && !onGrid(v));
        if (bad.length) add("space-scale", value);
      }
    }
  }
  if (small.length > SMALL_TEXT_BUDGET) out.push({ file, line: small[SMALL_TEXT_BUDGET], rule: "small-text", message: `${RULES["small-text"]} (${small.length} uses; the budget is ${SMALL_TEXT_BUDGET})` });
  return out;
}

/* ---------------- TS / TSX ---------------- */

const STYLE_PROPS = new Set(["fontSize", "borderRadius", "boxShadow", "backdropFilter", "textTransform", "letterSpacing", "transition", "fontFamily", "background", "backgroundImage", "color"]);

function lintTs(file, text, tokenFile, inDs) {
  const out = [];
  const sf = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true, file.endsWith("x") ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  const isTest = /\.test\.|\/tests?\//.test(file);
  const motionHome = MOTION_HOME.test(file);
  const add = (node, rule, extra = "") => {
    const { line } = sf.getLineAndCharacterOfPosition(node.getStart(sf));
    out.push({ file, line: line + 1, rule, message: `${RULES[rule]}${extra ? ` (${extra})` : ""}` });
  };
  const inType = (n) => {
    for (let p = n.parent; p; p = p.parent) {
      if (ts.isLiteralTypeNode(p) || ts.isImportDeclaration(p) || ts.isExportDeclaration(p) || ts.isImportTypeNode(p) || ts.isExternalModuleReference(p)) return true;
      if (ts.isCallExpression(p) && p.expression.kind === ts.SyntaxKind.ImportKeyword) return true;
      if (ts.isStatement(p) && !ts.isVariableStatement(p) && !ts.isReturnStatement(p) && !ts.isExpressionStatement(p)) break;
    }
    return false;
  };
  const isDirective = (n) => ts.isExpressionStatement(n.parent) && n.parent.parent === sf;
  const propName = (n) => (n && (ts.isIdentifier(n) || ts.isStringLiteral(n)) ? n.text : null);

  const system = SYSTEM_DIR.test(file);
  const primaries = [];
  const buttons = [];
  const visit = (node) => {
    if (!system && ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier) && /^d3-(scale|shape)$/.test(node.moduleSpecifier.text)) add(node, "raw-chart", node.moduleSpecifier.text);
    if (!system && (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node))) {
      const tag = node.tagName.getText(sf);
      if (tag === "button" || tag === "motion.button") buttons.push(node);
      if (tag === "Button") {
        const v = node.attributes.properties.find((a) => ts.isJsxAttribute(a) && a.name.getText(sf) === "variant");
        if (v && v.initializer && ts.isStringLiteral(v.initializer) && v.initializer.text === "primary") primaries.push(node);
      }
    }
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node) || ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node) || ts.isJsxText(node)) {
      if (!inType(node) && !isDirective(node)) {
        const s = node.text;
        const style = node.parent && ts.isPropertyAssignment(node.parent) && STYLE_PROPS.has(propName(node.parent.name));
        const hits = textChecks(s, { copy: !style, style, iso: !isTest });
        for (const h of hits) if (!(h === "colour-literal" && tokenFile)) add(node, h, JSON.stringify(s.length > 60 ? `${s.slice(0, 57)}…` : s));
      }
    }
    if (ts.isPropertyAssignment(node) || ts.isJsxAttribute(node)) {
      const name = ts.isJsxAttribute(node) ? node.name.getText(sf) : propName(node.name);
      let init = ts.isJsxAttribute(node) ? node.initializer : node.initializer;
      if (init && ts.isJsxExpression(init)) init = init.expression;
      const lit = init && (ts.isNumericLiteral(init) ? Number(init.text) : ts.isStringLiteral(init) || ts.isNoSubstitutionTemplateLiteral(init) ? init.text : null);
      if (name === "fontSize" || name === "font-size") {
        const px = typeof lit === "number" ? lit : typeof lit === "string" ? (pxValue(lit) ?? (/^\d+(\.\d+)?$/.test(lit) ? Number(lit) : null)) : null;
        if (px !== null && px < 11) add(node, "font-size", String(lit));
      }
      if (name === "backdropFilter") add(node, "decoration", name);
      if (!system && name === "borderRadius" && lit !== null && lit !== 0 && !(typeof lit === "string" && RADIUS_OK(lit))) add(node, "radius", String(lit));
      if (!system && name === "textTransform" && lit === "uppercase") add(node, "caps", "uppercase");
      if (name === "fontVariant" && typeof lit === "string" && /small-caps/.test(lit)) add(node, "caps", lit);
      if (name === "fontFamily" && typeof lit === "string" && !new RegExp(`^${FONT_VAR.source}$`).test(lit)) add(node, "font-family", lit);
      if (name === "transition" && typeof lit === "string" && durationsMs(lit).some((x) => x > 200)) add(node, "motion", lit);
      if (!motionHome && ts.isPropertyAssignment(node) && ts.isObjectLiteralExpression(node.parent)) {
        if ((name === "stiffness" || name === "damping" || name === "bounce") && typeof lit === "number") add(node, "motion-preset", name);
        if (name === "duration" && typeof lit === "number" && lit > 0.2 && lit < 30) add(node, "motion-preset", `duration ${lit}`);
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  for (const n of primaries.slice(1)) add(n, "one-primary", `${primaries.length} in this file`);
  if (buttons.length) add(buttons[0], "raw-button", `${buttons.length} in this file`);
  if (!inDs && /@keyframes/.test(text)) {
    const line = lineOf(text, text.indexOf("@keyframes"));
    out.push({ file, line, rule: "motion", message: `${RULES.motion} (@keyframes)` });
  }
  return out;
}

/** Lint one file's text. `file` is a repo-relative path with forward slashes. */
export function lintSource(file, text) {
  const f = file.split(sep).join("/");
  const tokenFile = TOKEN_FILES.some((r) => r.test(f));
  const inDs = DS_DIR.test(f);
  const out = [];
  const allow = allowances(text, f, out);
  const found = f.endsWith(".css") ? lintCss(f, text, tokenFile, inDs) : lintTs(f, text, tokenFile, inDs);
  for (const x of found) {
    const ok = allow.get(x.line);
    if (!(ok && ok.has(x.rule))) out.push(x);
  }
  for (const x of out) if (WARN_RULES.has(x.rule)) x.level = "warn";
  return out.sort((a, b) => a.line - b.line);
}

/** Files in scope under a web root. */
export function scopeFiles(root) {
  const walk = (dir) => {
    let names;
    try {
      names = readdirSync(dir);
    } catch {
      return [];
    }
    return names.flatMap((n) => {
      const p = join(dir, n);
      if (statSync(p).isDirectory()) return n === "node_modules" ? [] : walk(p);
      return /\.(css|tsx?)$/.test(n) && !/\.d\.ts$|\.test\.tsx?$/.test(n) ? [p] : [];
    });
  };
  const app = walk(join(root, "src", "app")).filter((p) => {
    const r = relative(root, p).split(sep).join("/");
    return r.includes("/(neritic)/") || r.startsWith("src/app/system/");
  });
  return [...walk(join(root, "src", "nx")), ...app];
}

/** Every finding, warnings included (each carries `level: "warn"`). */
export function lintAll(root) {
  return scopeFiles(root).flatMap((p) => lintSource(relative(root, p), readFileSync(p, "utf8")));
}

/** The findings that fail the run. */
export function lintRoot(root) {
  return lintAll(root).filter((x) => x.level !== "warn");
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const root = join(fileURLToPath(new URL(".", import.meta.url)), "..");
  const files = scopeFiles(root);
  const all = lintAll(root);
  const found = all.filter((x) => x.level !== "warn");
  const warned = all.filter((x) => x.level === "warn");
  for (const x of found) console.log(`${x.file}:${x.line}  ${x.rule}  ${x.message}`);
  if (process.argv.includes("--warnings")) for (const x of warned) console.log(`${x.file}:${x.line}  warning  ${x.rule}  ${x.message}`);
  if (found.length) {
    console.error(`\nlint:design: ${found.length} problem${found.length === 1 ? "" : "s"} in ${new Set(found.map((x) => x.file)).size} file(s) of ${files.length}. Rules: docs/coastwatch/v12/design-system/README.md section 3, v13 section 10, v14 section 8.`);
    process.exit(1);
  }
  const byRule = [...new Set(warned.map((x) => x.rule))].map((r) => `${r} ${warned.filter((x) => x.rule === r).length}`).join(", ");
  console.log(`lint:design: ${files.length} files clean.${warned.length ? ` ${warned.length} warning${warned.length === 1 ? "" : "s"} (${byRule}); list them with --warnings.` : ""}`);
}
