"use client";

import { useEffect, useRef } from "react";
import type { Manifest } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { PackDef } from "./packs";
import { OfflinePanel } from "./OfflinePanel";
import { getRun } from "./runStore";

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';

/**
 * The field-pack dialog, the same on Explore and My Coast: modal, focus moves into it and stays there
 * (Tab wraps), Escape closes it, and focus returns to what opened it. A tap on the backdrop closes it
 * too, except while a download runs (a stray tap must not look like cancelling one); closing never
 * stops a download, and reopening shows its progress.
 */
export function PackDialog({ pack, manifest, baseUrl, official, onClose }: { pack: PackDef; manifest: Manifest; baseUrl: string; official?: OfficialDataset | null; onClose: () => void }) {
  const box = useRef<HTMLDivElement | null>(null);
  const close = useRef(onClose);
  useEffect(() => {
    close.current = onClose;
  }, [onClose]);
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    box.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        close.current();
        return;
      }
      if (e.key !== "Tab" || !box.current) return;
      const els = [...box.current.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((el) => el.offsetParent !== null || el === document.activeElement);
      if (!els.length) return e.preventDefault();
      const first = els[0];
      const last = els[els.length - 1];
      const at = document.activeElement;
      if (e.shiftKey && (at === first || at === box.current || !box.current.contains(at))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (at === last || !box.current.contains(at))) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      // back to the control that opened the dialog, when it is still on the page
      if (opener && opener.isConnected) opener.focus();
    };
  }, []);
  return (
    <div
      className="nx-veil fixed inset-0 z-50 grid place-items-center bg-[rgba(4,16,22,0.55)] p-3"
      onClick={(e) => e.target === e.currentTarget && getRun(pack.id).state !== "downloading" && onClose()}
      data-testid="pack-dialog-backdrop"
    >
      <div ref={box} tabIndex={-1} role="dialog" aria-modal="true" aria-label={`Field pack: ${pack.name}`} className="env-chart nx-deck nx-rise max-h-[92dvh] w-[min(520px,100%)] overflow-y-auto p-5 text-n-ink outline-none" data-testid="pack-dialog">
        <OfflinePanel pack={pack} manifest={manifest} official={official} baseUrl={baseUrl} onClose={onClose} />
      </div>
    </div>
  );
}
