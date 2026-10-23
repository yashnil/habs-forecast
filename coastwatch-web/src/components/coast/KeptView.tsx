"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { KEPT_WARNING, stamp, deleteKept, keptFilename, keptHtml, keptSections, keptTitle, readKept, type Kept } from "@/lib/keptBriefing";
import { coastHref, parseAudience } from "@/lib/coast";
import { AGENCY_LINKS } from "@/content/coast";
import { BrandMark } from "@/components/brand/BrandMark";

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

  const offline = typeof navigator !== "undefined" && !navigator.onLine;
  const tel = "nx-focus inline-flex min-h-11 items-center font-data text-[14px] font-light text-n-ink underline decoration-n-official-line underline-offset-4 print:text-black";
  return (
    <div className="env-chart min-h-dvh bg-n-bg text-n-ink print:bg-white print:text-black">
      <main id="main" className="mx-auto w-full max-w-[720px] px-4 pb-16 pt-5 print:max-w-none print:p-0" data-testid="kept-page">
        <p className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 print:hidden">
          <Link href="/" className="nx-focus inline-flex min-h-11 items-center gap-2 rounded-[3px] text-[17px] font-semibold tracking-[-0.01em] text-n-ink" aria-label="Neritic home">
            <BrandMark size={26} /> Neritic
          </Link>
          <Link href={k ? coastHref({ place: k.snapshot.place.key, for: parseAudience(k.snapshot.audience), species: k.snapshot.species.id }) : "/coast"} className="nx-focus inline-flex min-h-11 items-center gap-1 text-[14px] text-n-ink underline decoration-n-line-strong underline-offset-4">
            {offline ? "My Coast (needs a connection)" : "Current information on My Coast"}
          </Link>
        </p>
        {!k ? (
          <p className="mt-8 text-[16px] text-n-ink-2" data-testid="kept-empty">
            No briefing is kept on this device. On My Coast, choose &ldquo;Keep a dated copy&rdquo;.
          </p>
        ) : (
          <article data-testid="kept-article" className="mt-6">
            <p className="nx-caps text-n-ink-2 print:text-black">Neritic briefing · kept copy{offline ? " · read offline" : ""}</p>
            <h1 className="mt-1 font-serif text-[30px] font-medium leading-[1.1] tracking-[-0.015em] sm:text-[38px]" data-testid="kept-title">
              {keptTitle(k)}
            </h1>
            <p className="mt-2 font-data text-[14px] font-light text-n-ink-2 print:text-black" data-testid="kept-saved">
              Saved {stamp(k.keptAt)}
              {k.snapshot.place.county ? ` · ${k.snapshot.place.county} County` : ""}
            </p>
            <p className="mt-4 border-l-2 border-n-official py-1 pl-4 text-[15px] font-medium leading-snug print:border-black" data-testid="kept-warning">
              {KEPT_WARNING}
            </p>
            <div className="mt-2 flex flex-wrap gap-x-5 print:hidden">
              <button type="button" onClick={() => window.print()} className="nx-press nx-focus inline-flex min-h-11 items-center text-[14.5px] font-medium text-n-ink underline decoration-n-line-strong underline-offset-4" data-testid="kept-print">
                Print
              </button>
              <button type="button" onClick={() => download(k)} className="nx-press nx-focus inline-flex min-h-11 items-center text-[14.5px] font-medium text-n-ink underline decoration-n-line-strong underline-offset-4" data-testid="kept-file">
                Save as a file (opens without a connection)
              </button>
              <button
                type="button"
                onClick={() => {
                  deleteKept(k.id);
                  setList(readKept());
                }}
                className="nx-press nx-focus inline-flex min-h-11 items-center text-[14.5px] text-n-ink-3 underline decoration-n-line underline-offset-4"
                data-testid="kept-delete"
              >
                Delete this copy
              </button>
            </div>
            {keptSections(k).map((s) => (
              <section key={s.heading} className="mt-7 break-inside-avoid">
                <h2 className="nx-caps border-b border-n-line-strong pb-1.5 text-n-ink-2 print:border-black print:text-black">{s.heading}</h2>
                <ul>
                  {s.lines.map((l, i) => {
                    const phone = /^(.*): (\(?\d[\d ()-]{8,})$/.exec(l.text);
                    return (
                      <li key={i} className="border-b border-n-line py-2.5 print:border-gray-300">
                        {phone ? (
                          <span className="block text-[15.5px]">
                            {phone[1]}:{" "}
                            <a href={`tel:+${phone[2].replace(/\D/g, "").replace(/^(?!1)/, "1")}`} className={tel}>
                              {phone[2]}
                            </a>
                          </span>
                        ) : (
                          <span className="block text-[15.5px] leading-snug">{l.text}</span>
                        )}
                        {l.note ? <span className="mt-0.5 block text-[13.5px] text-n-ink-2 print:text-black">{l.note}</span> : null}
                        {l.url ? (
                          <a href={l.url} target="_blank" rel="noopener noreferrer" className="nx-focus mt-0.5 block break-all text-[13px] text-n-accent print:text-black">
                            {l.url}
                          </a>
                        ) : null}
                      </li>
                    );
                  })}
                </ul>
              </section>
            ))}
            <section className="mt-7 border-l-2 border-n-official pl-4 print:border-black" data-testid="kept-agencies">
              <p className="nx-caps text-n-official print:text-black">Official notices · CDFW and CDPH</p>
              <p className="mt-1 text-[14.5px] leading-snug">No notice listed does not mean open or safe. The agencies are the authority; call before you go.</p>
              <p className="flex flex-wrap gap-x-5">
                <a href={`tel:${AGENCY_LINKS.cdfwClosureLine.tel}`} className={tel} data-testid="kept-tel-cdfw">
                  CDFW {AGENCY_LINKS.cdfwClosureLine.phone}
                </a>
                <a href={`tel:${AGENCY_LINKS.cdphLine.tel}`} className={tel} data-testid="kept-tel-cdph">
                  CDPH {AGENCY_LINKS.cdphLine.phone}
                </a>
              </p>
            </section>
            <p className="nx-margin mt-6 text-[13px] text-n-ink-2 print:text-black">
              Built from Neritic data generated {stamp(k.snapshot.dataGeneratedAt)}. Neritic is an independent research project, not a government agency.
            </p>
          </article>
        )}
        {list.length > 1 ? (
          <nav className="mt-10 print:hidden" aria-label="Other kept briefings">
            <h2 className="nx-caps text-n-ink-2">Other kept copies on this device</h2>
            <ul>
              {list
                .filter((x) => x.id !== k?.id)
                .map((x) => (
                  <li key={x.id}>
                    <Link href={`/coast/kept?id=${encodeURIComponent(x.id)}`} className="nx-focus inline-flex min-h-11 items-center text-[14.5px] text-n-ink underline decoration-n-line-strong underline-offset-4">
                      {keptTitle(x)} · saved {stamp(x.keptAt)}
                    </Link>
                  </li>
                ))}
            </ul>
          </nav>
        ) : null}
      </main>
    </div>
  );
}
