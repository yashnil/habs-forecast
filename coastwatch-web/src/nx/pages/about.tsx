import Link from "next/link";
import { ABOUT, RESEARCH } from "@/content/copy";
import { Glyph, Micro, type GlyphKind } from "@/nx/ds/synth/Grammar";
import { Button } from "@/nx/ds/Button";
import s from "./about.module.css";

/** Each kind of information by its glyph in the v14 grammar (the v4 page drew a colour chip). */
const GLYPH: Record<string, GlyphKind> = { official: "unverified-coral", model: "predicted", measured: "measured", historical: "old" };
const WORD: Record<string, string> = { official: "Official", model: "Model", measured: "Measured", historical: "Historical" };

/** About's four kinds of information, word for word (copy.ts ABOUT.kinds), each with where it lives. */
export function AboutKinds() {
  return (
    <dl className={s.kinds} data-testid="about-kinds">
      {ABOUT.kinds.map((k) => (
        <div key={k.kind} className={s.kind}>
          <dt>
            <Micro glyph={GLYPH[k.kind]}>{WORD[k.kind]}</Micro>
            <span className={s.kindName}>{k.name}</span>
          </dt>
          <dd>
            {k.text}{" "}
            <Link href={k.href} className={s.kindLink}>
              {k.where} →
            </Link>
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** The publication, as a citation with the one action (read it at its DOI). */
export function Publication() {
  return (
    <figure className={s.pub} data-testid="publication">
      <Micro as="p">Publication</Micro>
      <p className={s.pubTitle}>{RESEARCH.title}</p>
      <p className={s.pubMeta}>
        {RESEARCH.author} · {RESEARCH.venue}
        <br />
        {RESEARCH.published}
      </p>
      <p className={s.pubAct}>
        <Button variant="primary" href={RESEARCH.doi} external data-testid="publication-doi">
          Read the paper
        </Button>
        <span className={s.doi}>doi:10.33422/ccgconf.v2i2.1619</span>
      </p>
    </figure>
  );
}

/** The data providers, one row each (copy.ts ABOUT.providers). */
export function Providers() {
  return (
    <ul className={s.providers} data-testid="about-providers">
      {ABOUT.providers.map((p) => (
        <li key={p.name}>
          <span className={s.provName}>{p.name}</span>
          <span className={s.provWhat}>{p.what}</span>
        </li>
      ))}
    </ul>
  );
}
