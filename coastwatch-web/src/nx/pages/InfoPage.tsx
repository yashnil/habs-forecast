import Link from "next/link";
import type { ReactNode } from "react";
import { BRAND } from "@/content/brand";
import { RESEARCH } from "@/content/copy";
import { DATA_UNAVAILABLE, LEGAL_FOOTER, SITE_FOOTER } from "@/content/pages";
import { PageLayout } from "@/nx/ds/layout/Desk";
import { Micro } from "@/nx/ds/synth/Grammar";
import { cx } from "@/nx/ds/util";
import { Contents, type ContentsItem } from "./Contents";
import { RegistryNote, type RegistryTrust } from "./RegistryNote";
import s from "./info.module.css";
import "./infoShell.css";

export type { ContentsItem } from "./Contents";
export type { RegistryTrust } from "./RegistryNote";

export interface InfoPageProps {
  /** Names the page for the stylesheet and tests ("about", "privacy"). */
  id: string;
  /** The article's own test id, kept from the page's earlier frame ("about-page", "privacy-page"). */
  testId?: string;
  /** One micro line over the title ("About Neritic"). */
  kicker?: ReactNode;
  /** The page's h1. Omitted: the body brings its own (Fisheries' story keeps its headline); the registry line and the
   *  neatline still open the page. */
  title?: ReactNode;
  titleProps?: { [key: `data-${string}`]: string };
  lede?: ReactNode;
  /** Under the lede: an effective date, a draft banner, a link. */
  head?: ReactNode;
  /** The official-notice registry line (every informational page carries it). `undefined`: not shown (error pages). */
  trust?: RegistryTrust | null;
  /** The in-page contents, for a long page: the ids of its h2s, in order. */
  contents?: ContentsItem[];
  /** "read" (default): one column at a reading measure; "wide": the body takes the sheet (a figure-led page). */
  width?: "read" | "wide";
  /** Which standing disclaimer closes the page: the reference pages' (default) or the legal pages'. */
  footer?: "site" | "legal" | "none";
  /** The body is long reading (h2, p, ul): the prose styles apply. */
  prose?: boolean;
  children: ReactNode;
}

/** The chart's neatline: a heavy rule, a hairline, and minute ticks between them (every fifth longer). */
function Neatline() {
  return (
    <div className={s.neat} aria-hidden="true">
      <svg preserveAspectRatio="none">
        <defs>
          <pattern id="nxp-ticks" width="60" height="12" patternUnits="userSpaceOnUse">
            <path d="M0.5 1V9M12.5 1V5M24.5 1V5M36.5 1V5M48.5 1V5" stroke="currentColor" strokeWidth="1" />
          </pattern>
        </defs>
        <line x1="0" x2="100%" y1="0.75" y2="0.75" className={s.neatRule} />
        <rect x="0" y="0" width="100%" height="12" fill="url(#nxp-ticks)" />
        <line x1="0" x2="100%" y1="11.5" y2="11.5" className={s.neatHair} />
      </svg>
    </div>
  );
}

/** The links every informational page ends with: the reference pages, the paper, and the three legal pages. */
const FOOT_LINKS: Array<[string, string, boolean?]> = [
  ["/about", "About"],
  ["/sources", "Data & sources"],
  [RESEARCH.doi, "Research paper ↗", true],
  ["/privacy", "Privacy"],
  ["/terms", "Terms"],
  ["/sms-terms", "SMS terms"],
];

function InfoFooter({ kind }: { kind: "site" | "legal" }) {
  return (
    <footer className={s.foot} data-testid={kind === "legal" ? "pilot-footer" : "site-footer"}>
      <p className={s.footBrand} data-testid="footer-brand">
        {SITE_FOOTER.brand}
      </p>
      <p className={s.footText} data-testid="footer-disclaimer">
        {kind === "legal" ? LEGAL_FOOTER : SITE_FOOTER.disclaimer}
      </p>
      <p className={s.footText} data-testid="footer-independence">
        {BRAND.independence}
      </p>
      <nav aria-label={`About ${BRAND.name}`} className={s.footNav}>
        {FOOT_LINKS.map(([href, label, external]) =>
          external ? (
            <a key={href} href={href} target="_blank" rel="noreferrer">
              {label}
            </a>
          ) : (
            <Link key={href} href={href} data-testid={`footer-${href.slice(1)}`}>
              {label}
            </Link>
          ),
        )}
      </nav>
    </footer>
  );
}

/**
 * An informational page (v15.1, docs/coastwatch/v15.1/routes.md): About, Sources, the legal pages, Fisheries, kept
 * briefings and the error pages, in the shell every other page uses (the bar, the menu, the tab bar) and on the one
 * chart, which stays mounted and shows as a quiet edge beside the sheet (infoShell.css). It declares the desk layout,
 * so the phone reads it as a page rather than a sheet; the head (micro kicker, title, lede, the registry line) closes
 * with the chart's neatline; long pages list their sections beside the text. Usable from server and client pages.
 */
export function InfoPage({ id, testId, kicker, title, titleProps, lede, head, trust, contents, width = "read", footer = "site", prose = false, children }: InfoPageProps) {
  const toc = contents && contents.length > 2 ? contents : null;
  return (
    <>
      <PageLayout mode="desk" />
      <span hidden data-nx-info={id} />
      <article className={cx(s.page, width === "wide" && s.wide)} data-testid="nx-info-page" data-page={id}>
        <div data-testid={testId} className={s.inner}>
          <header className={cx(s.head, !title && s.headBare)}>
            {kicker ? (
              <Micro as="p" className={s.kicker}>
                {kicker}
              </Micro>
            ) : null}
            {title ? (
              <h1 className={s.title} {...titleProps}>
                {title}
              </h1>
            ) : null}
            {lede ? <p className={s.lede}>{lede}</p> : null}
            {head ? <div className={s.headMore}>{head}</div> : null}
            {trust !== undefined ? (
              <div className={s.reg}>
                {trust && !trust.generatedAt ? (
                  <p role="alert" className={cx(s.note, s.noteReg)} data-testid="data-unavailable">
                    {DATA_UNAVAILABLE}
                  </p>
                ) : null}
                <RegistryNote trust={trust} />
              </div>
            ) : null}
            <Neatline />
          </header>
          <div className={cx(s.body, toc && s.withToc)}>
            {toc ? <Contents items={toc} /> : null}
            <div className={cx(s.main, prose && s.prose)}>{children}</div>
          </div>
        </div>
        {footer !== "none" ? <InfoFooter kind={footer} /> : null}
      </article>
    </>
  );
}

/** A section of a reference page: an h2 (its id is the contents' anchor) and its work. */
export function InfoSection({ id, title, intro, children, ...rest }: { id: string; title: ReactNode; intro?: ReactNode; children?: ReactNode; [key: `data-${string}`]: string | undefined }) {
  return (
    <section className={s.section} aria-labelledby={id} {...rest}>
      <h2 id={id}>{title}</h2>
      {intro ? <p className={s.sectionIntro}>{intro}</p> : null}
      {children}
    </section>
  );
}

export { s as infoStyles };
