"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { KEPT_WARNING, stamp, deleteKept, keptFilename, keptHtml, keptSections, keptTitle, readKept, type Kept } from "@/lib/keptBriefing";
import { coastHref, parseAudience } from "@/lib/coast";

/** Kept briefings on this device; one is shown in full, ready to print or save as a file. */
export function KeptView() {
  return (
    <Suspense fallback={null}>
      <KeptViewInner />
    </Suspense>
  );
}

function KeptViewInner() {
  const id = useSearchParams().get("id");
  const [list, setList] = useState<Kept[] | null>(null);
  useEffect(() => setList(readKept()), []);
  if (list === null) return null;
  const k = list.find((x) => x.id === id) ?? list[0] ?? null;

  const download = (x: Kept) => {
    const url = URL.createObjectURL(new Blob([keptHtml(x)], { type: "text/html" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = keptFilename(x);
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <main className="mx-auto min-h-screen w-full max-w-[720px] bg-surface px-4 py-6 text-ink print:max-w-none print:p-0" data-testid="kept-page">
      <p className="text-[14px] text-ink-2 print:hidden">
        <Link href={k ? coastHref({ place: k.snapshot.place.key, for: parseAudience(k.snapshot.audience), species: k.snapshot.species.id }) : "/coast"} className="inline-flex min-h-[44px] items-center font-medium text-accent">
          ← Current information on My Coast
        </Link>
      </p>
      {!k ? (
        <p className="mt-4 text-[16px]" data-testid="kept-empty">
          No briefing is kept on this device. On My Coast, choose &ldquo;Keep a dated copy&rdquo;.
        </p>
      ) : (
        <article data-testid="kept-article">
          <p className="text-[14px] text-ink-2">Neritic briefing, kept copy</p>
          <h1 className="mt-1 font-display text-[24px] font-medium leading-snug" data-testid="kept-title">
            {keptTitle(k)}
          </h1>
          <p className="mt-1 text-[15px] text-ink-2" data-testid="kept-saved">
            Saved {stamp(k.keptAt)}
            {k.snapshot.place.county ? ` · ${k.snapshot.place.county} County` : ""}
          </p>
          <p className="mt-3 rounded-lg border-2 border-ink px-3 py-2 text-[15px] font-semibold" data-testid="kept-warning">
            {KEPT_WARNING}
          </p>
          <div className="mt-2 flex flex-wrap gap-x-5 print:hidden">
            <button type="button" onClick={() => window.print()} className="cw-press min-h-[44px] text-[15px] font-medium text-accent" data-testid="kept-print">
              Print
            </button>
            <button type="button" onClick={() => download(k)} className="cw-press min-h-[44px] text-[15px] font-medium text-accent" data-testid="kept-file">
              Save as a file (opens without a connection)
            </button>
            <button
              type="button"
              onClick={() => {
                deleteKept(k.id);
                setList(readKept());
              }}
              className="cw-press min-h-[44px] text-[15px] text-ink-3 underline underline-offset-2"
              data-testid="kept-delete"
            >
              Delete this copy
            </button>
          </div>
          {keptSections(k).map((s) => (
            <section key={s.heading} className="mt-5 break-inside-avoid">
              <h2 className="border-b-2 border-ink pb-0.5 text-[17px] font-semibold">{s.heading}</h2>
              <ul>
                {s.lines.map((l, i) => (
                  <li key={i} className="border-b border-hairline py-2">
                    <span className="block text-[16px]">{l.text}</span>
                    {l.note ? <span className="block text-[14px] text-ink-2">{l.note}</span> : null}
                    {l.url ? (
                      <a href={l.url} target="_blank" rel="noopener noreferrer" className="block break-all text-[13px] text-accent print:text-ink">
                        {l.url}
                      </a>
                    ) : null}
                  </li>
                ))}
              </ul>
            </section>
          ))}
          <p className="mt-5 text-[13px] text-ink-2">
            Built from Neritic data generated {stamp(k.snapshot.dataGeneratedAt)}. Neritic is an independent research project, not a government agency.
          </p>
        </article>
      )}
      {list.length > 1 ? (
        <nav className="mt-8 print:hidden" aria-label="Other kept briefings">
          <h2 className="text-[15px] font-semibold">Other kept copies on this device</h2>
          <ul>
            {list
              .filter((x) => x.id !== k?.id)
              .map((x) => (
                <li key={x.id}>
                  <Link href={`/coast/kept?id=${encodeURIComponent(x.id)}`} className="inline-flex min-h-[44px] items-center text-[15px] text-accent">
                    {keptTitle(x)} · saved {stamp(x.keptAt)}
                  </Link>
                </li>
              ))}
          </ul>
        </nav>
      ) : null}
    </main>
  );
}
