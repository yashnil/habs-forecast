import Link from "next/link";
import { BRAND } from "@/content/brand";
import { AppShell } from "@/components/AppShell";
import { KindChip, type Kind } from "@/components/ui/Kinds";
import { PurposeSwitch } from "@/components/welcome/PurposeSwitch";
import { ABOUT, RESEARCH } from "@/content/copy";
import { loadData } from "@/lib/data";
import { sourceStatus } from "@/lib/layers";

export const dynamic = "force-dynamic";
export const metadata = { title: "About" };

/**
 * What Neritic is, the kinds of information it shows and where, what it cannot tell you, the
 * research it comes from (and what that research does not cover), and its data providers.
 */
export default async function AboutPage() {
  const data = await loadData();
  const manifest = data.ok ? data.manifest : null;
  return (
    <AppShell
      active="about"
      manifest={manifest}
      official={data.ok ? data.official : null}
      officialStatus={data.ok ? sourceStatus(data.manifest, "official") : null}
      scroll="page"
    >
      <article className="mx-auto w-full max-w-[1120px] px-4 pb-20 pt-8 sm:px-8 sm:pt-12" data-testid="about-page">
        <header className="max-w-[760px] space-y-4">
          <p className="text-[13px] font-medium text-ink-3">About Neritic · {BRAND.tagline}</p>
          <h1 className="text-balance font-display text-[34px] font-medium leading-[1.08] text-ink sm:text-[46px]">{ABOUT.title}</h1>
          <p className="font-display text-[19px] leading-[1.5] text-ink-2 sm:text-[21px]">{ABOUT.lede}</p>
          <p className="text-[16px]">
            <Link href="/" className="font-medium text-accent underline decoration-accent/35 underline-offset-2 hover:decoration-accent" data-testid="about-tour">
              Open Explore, the map →
            </Link>
          </p>
          <p className="max-w-[68ch] text-[14px] leading-relaxed text-ink-3" data-testid="about-independence">
            {BRAND.independence} {BRAND.formerly}
          </p>
        </header>

        <section aria-labelledby="purpose-h" className="mt-10 max-w-[760px] space-y-2" data-testid="about-purpose">
          <h2 id="purpose-h" className="text-[13.5px] font-semibold text-ink-3">
            What Neritic is set up for on this device
          </h2>
          <PurposeSwitch showReset />
          <p className="text-[14px] text-ink-3">It changes only where you start and what comes first. Every page, value, caveat and official notice is the same for everyone. Nothing is sent anywhere.</p>
        </section>

        <section aria-labelledby="kinds-h" className="mt-12 border-t border-hairline pt-8">
          <h2 id="kinds-h" className="font-display text-[26px] font-medium leading-tight text-ink">
            Four kinds of information, never mixed
          </h2>
          <p className="mt-2 max-w-[68ch] text-[16px] leading-relaxed text-ink-2">{ABOUT.kindsIntro}</p>
          <dl className="mt-6 grid gap-x-10 gap-y-7 md:grid-cols-2" data-testid="about-kinds">
            {ABOUT.kinds.map((k) => (
              <div key={k.kind} className="space-y-1.5">
                <dt className="flex flex-wrap items-center gap-2">
                  <KindChip kind={k.kind as Kind} />
                  <span className="text-[16px] font-semibold text-ink">{k.name}</span>
                </dt>
                <dd className="text-[15px] leading-relaxed text-ink-2">
                  {k.text}{" "}
                  <Link href={k.href} className="font-medium text-accent underline-offset-2 hover:underline">
                    {k.where} →
                  </Link>
                </dd>
              </div>
            ))}
          </dl>
        </section>

        <section aria-labelledby="cannot-h" className="mt-12 grid gap-x-12 gap-y-4 border-t border-hairline pt-8 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]" data-testid="about-cannot">
          <h2 id="cannot-h" className="font-display text-[26px] font-medium leading-tight text-ink">
            What Neritic cannot tell you
          </h2>
          <ul className="space-y-3 text-[16px] leading-relaxed text-ink-2">
            {ABOUT.cannot.map((t) => (
              <li key={t} className="border-l-2 border-official-line pl-3">
                {t}
              </li>
            ))}
          </ul>
        </section>

        <section aria-labelledby="research-h" className="mt-12 border-t border-hairline pt-8" data-testid="research">
          <h2 id="research-h" className="font-display text-[26px] font-medium leading-tight text-ink">
            The research behind it
          </h2>
          <div className="mt-5 grid gap-x-12 gap-y-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
            <figure className="rounded-2xl border border-hairline bg-surface p-6 sm:p-7" data-testid="publication">
              <p className="text-[13px] font-medium text-ink-3">Publication</p>
              <p className="mt-2 font-display text-[22px] font-medium leading-snug text-ink sm:text-[24px]">{RESEARCH.title}</p>
              <p className="mt-3 text-[14px] leading-relaxed text-ink-2">
                {RESEARCH.author} · {RESEARCH.venue}
                <br />
                {RESEARCH.published}
              </p>
              <p className="mt-4 flex flex-wrap items-center gap-3">
                <a
                  href={RESEARCH.doi}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 rounded-lg bg-navy-900 px-4 py-2.5 text-[15px] font-medium text-white hover:bg-navy-800"
                  data-testid="publication-doi"
                >
                  Read the paper <span aria-hidden>↗</span>
                </a>
                <span className="font-mono text-[13px] text-ink-3">doi:10.33422/ccgconf.v2i2.1619</span>
              </p>
            </figure>
            <div className="space-y-3 text-[15.5px] leading-relaxed text-ink-2">
              <p>{ABOUT.research.what}</p>
              <p>{ABOUT.research.result}</p>
              <p className="border-l-2 border-hairline-strong pl-3 text-ink" data-testid="research-relation">
                {ABOUT.research.relation}
              </p>
            </div>
          </div>
        </section>

        <section aria-labelledby="providers-h" className="mt-12 grid gap-x-12 gap-y-4 border-t border-hairline pt-8 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
          <div className="space-y-2">
            <h2 id="providers-h" className="font-display text-[26px] font-medium leading-tight text-ink">
              Data and methods
            </h2>
            <p className="text-[15px] leading-relaxed text-ink-2">{ABOUT.providersIntro}</p>
            <p className="flex flex-col gap-1.5 pt-1 text-[15px]">
              <Link href="/sources" className="font-medium text-accent underline-offset-2 hover:underline" data-testid="about-sources">
                Data status and licences →
              </Link>
              <a href={ABOUT.repo} target="_blank" rel="noreferrer" className="font-medium text-accent underline-offset-2 hover:underline">
                Code, methods and documentation ↗
              </a>
            </p>
          </div>
          <ul className="divide-y divide-hairline border-y border-hairline text-[15px]" data-testid="about-providers">
            {ABOUT.providers.map((p) => (
              <li key={p.name} className="grid gap-x-6 gap-y-0.5 py-2.5 sm:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
                <span className="font-medium text-ink">{p.name}</span>
                <span className="text-ink-2">{p.what}</span>
              </li>
            ))}
          </ul>
        </section>

        <section aria-labelledby="review-h" className="mt-12 border-t border-hairline pt-8">
          <h2 id="review-h" className="font-display text-[26px] font-medium leading-tight text-ink">
            Review status
          </h2>
          <ul className="mt-3 max-w-[68ch] list-disc space-y-1.5 pl-5 text-[16px] leading-relaxed text-ink-2" data-testid="about-review">
            {ABOUT.review.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </section>
      </article>
    </AppShell>
  );
}
