"use client";

import type { Association } from "@/lib/applicability";
import type { EvidenceNode } from "@/lib/harborEvidence";
import { AGENCY_LINKS } from "@/content/coast";
import { speciesPhrase, type Species } from "@/content/species";
import { dayLabel } from "@/v4/ui/time";
import { Icon } from "@/v4/ui/Icon";
import { Sheet } from "@/v4/motion/Sheet";

const ext = { target: "_blank", rel: "noreferrer" } as const;

/**
 * The notices tied to these waters (v11): the map's own detail. Hovering or focusing a notice lights its
 * line on the map; a tap on a line opens this sheet at that notice. Every word of the old list stays here.
 */
export function NoticesSheet({ open, onClose, focus, onLight, nodes, other, species, unsettled }: { open: boolean; onClose: () => void; focus: string[] | null; onLight: (ids: string[]) => void; nodes: EvidenceNode[]; other: Association[]; species: Species; unsettled: number }) {
  const onRow = (id: string) => ({ onPointerEnter: () => onLight([id]), onPointerLeave: () => onLight(focus ?? []), onFocus: () => onLight([id]), onBlur: () => onLight(focus ?? []) });
  const isFocus = (id: string) => !!focus?.includes(id);
  const rows = nodes.length + other.length;
  return (
    <Sheet open={open} onClose={onClose} modal={false} title="Notices on these waters" testid="notices-sheet" closeLabel="Close notices">
      <div className="flex flex-col gap-3 pb-6" data-testid="brief-official">
        <p className="text-[14px] leading-[20px] text-n-ink-2">Tap a line on the map, or a notice here, to light its limit.</p>
        <h3 id="official-h" className="nx-caps scroll-mt-16">
          Official notices · not verified
        </h3>
        {rows ? (
          <ul className="flex flex-col">
            {nodes.map((n) => (
              <li key={n.id} data-testid={`notice-${n.id}`} data-on={isFocus(n.id?.replace("official:", "") ?? "") || undefined} tabIndex={0} {...onRow(n.id?.replace("official:", "") ?? "")} className="coast-notice nx-focus">
                <p className="text-[15px] font-medium leading-snug">{n.text}</p>
                <p className="mt-0.5 text-[13.5px] text-n-ink-2">
                  {n.source.name}
                  {n.where?.name ? ` · ${n.where.name}` : ""}
                  {(n.time as { issued?: string | null }).issued ? ` · since ${dayLabel((n.time as { issued: string }).issued)}` : ""}
                  <span className="text-n-official"> · Not verified</span>
                </p>
                {n.freshness?.note && <p className="mt-1 text-[13.5px] text-n-official">{n.freshness.note}</p>}
                {n.source.url && (
                  <a href={n.source.url} {...ext} className="nx-link nx-focus inline-flex min-h-11 items-center gap-1 text-[13.5px]">
                    {n.source.name} notice <Icon name="external" className="h-3.5 w-3.5" />
                  </a>
                )}
              </li>
            ))}
            {other.length > 0 && (
              <li className="pt-3" data-testid="notices-other">
                <p className="nx-caps">{species.id === "any" ? "Also tied to these waters" : `Tied to these waters, not naming ${speciesPhrase(species)}`}</p>
                <ul className="flex flex-col">
                  {other.map((a) => {
                    const url = a.record.sources.find((x) => x.url.startsWith("https://"))?.url ?? null;
                    return (
                      <li key={a.record.id} data-testid={`notice-official:${a.record.id}`} data-on={isFocus(a.record.id) || undefined} tabIndex={0} {...onRow(a.record.id)} className="coast-notice nx-focus">
                        <p className="text-[14.5px] leading-snug">{a.record.title}</p>
                        <p className="flex flex-wrap items-center gap-x-3 text-[13.5px] text-n-ink-2">
                          <span>
                            {a.record.agency}
                            {a.record.effective_date ? ` · since ${dayLabel(a.record.effective_date)}` : ""}
                          </span>
                          <span className="text-n-official">Not verified</span>
                          {url && (
                            <a href={url} {...ext} className="nx-link nx-focus inline-flex min-h-11 items-center gap-1">
                              {a.record.agency} notice <Icon name="external" className="h-3.5 w-3.5" />
                            </a>
                          )}
                        </p>
                      </li>
                    );
                  })}
                </ul>
              </li>
            )}
          </ul>
        ) : (
          <p className="text-[14.5px] text-n-ink-2">No notice in Neritic&rsquo;s list is tied to these waters.</p>
        )}
        <p className="text-[13.5px] leading-[20px] text-n-ink-2" data-testid="official-caveat">
          <span className="text-n-official">Not verified</span> · may be incomplete · no notice listed does not mean open or safe{unsettled ? ` · ${unsettled} more unclear` : ""}. The agency page is the authority:{" "}
          <a href={`tel:${AGENCY_LINKS.cdfwClosureLine.tel}`} className="nx-link nx-focus whitespace-nowrap">
            CDFW {AGENCY_LINKS.cdfwClosureLine.phone}
          </a>
          {" · "}
          <a href={`tel:${AGENCY_LINKS.cdphLine.tel}`} className="nx-link nx-focus whitespace-nowrap">
            CDPH {AGENCY_LINKS.cdphLine.phone}
          </a>
        </p>
      </div>
    </Sheet>
  );
}
