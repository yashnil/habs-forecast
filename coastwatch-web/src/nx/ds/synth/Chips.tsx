"use client";

import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { MAP_LAYERS, useMapLayers, type MapLayerDef } from "@/nx/chart/mapLayers";
import { cx } from "../util";
import s from "./synth.module.css";

export interface ChipProps {
  pressed: boolean;
  onPress: () => void;
  children: ReactNode;
  /** a swatch drawn before the words (the layer's own mark) */
  swatch?: MapLayerDef["swatch"];
  /** the one-line key, shown on hover or focus (and read as the chip's description) */
  keyText?: { question: string; key: string };
  className?: string;
  "data-testid"?: string;
  "data-layer"?: string;
}

/** A pressable chip: words with a swatch; pressed is an ultramarine wash. Its key opens on hover or focus. */
export function Chip({ pressed, onPress, children, swatch, keyText, className, "data-testid": testid, "data-layer": layer }: ChipProps) {
  const id = useId();
  return (
    <button type="button" className={cx(s.ly, className)} aria-pressed={pressed} aria-describedby={keyText ? id : undefined} onClick={onPress} data-testid={testid} data-layer={layer}>
      {swatch ? <i className={cx(s.lySw, s[`sw-${swatch}`])} aria-hidden="true" /> : null}
      <span className={s.lyT}>{children}</span>
      {keyText ? (
        <span className={s.lyKey} id={id} role="tooltip">
          <b>{keyText.question}</b>
          {keyText.key}
        </span>
      ) : null}
    </button>
  );
}

const LayersIcon = () => (
  <svg viewBox="0 0 20 20" aria-hidden="true">
    <path d="M10 2.5l7.5 4-7.5 4-7.5-4z" />
    <path d="M2.5 10.2l7.5 4 7.5-4M2.5 13.8l7.5 4 7.5-4" />
  </svg>
);

/**
 * The layer chips (v14 README §4): Regulation · Depth (fm) · Sea now · Toxin tests, and any data layers the page adds.
 * The row is the legend and the control. A wide screen shows the row under the brand, first in tab order after the
 * nav; a phone shows one "Layers" button that opens the list, each pressed layer with its key under its name.
 */
export function LayerChips({ className }: { className?: string }) {
  const { on, layers, toggle } = useMapLayers();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const listId = useId();
  useEffect(() => {
    if (!open) return;
    const off = (e: PointerEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    const key = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", off);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("pointerdown", off);
      document.removeEventListener("keydown", key);
    };
  }, [open]);
  const n = layers.filter((l) => on[l.id]).length;
  return (
    <div ref={ref} className={cx(s.layers, open && s.layersOpen, className)} data-testid="nx-layers" data-n={n}>
      <button type="button" className={s.lyHead} aria-label="Layers" aria-expanded={open} aria-controls={listId} onClick={() => setOpen((o) => !o)} data-testid="nx-layers-button">
        <LayersIcon />
        <span>Layers</span>
      </button>
      {/* the group is the chips themselves: on a phone it is hidden until "Layers" opens it */}
      <div className={s.lyList} id={listId} role="group" aria-label="Map layers">
        {layers.map((l) => (
          <Chip key={l.id} pressed={!!on[l.id]} onPress={() => toggle(l.id)} swatch={l.swatch} keyText={{ question: l.question, key: l.key }} data-testid={`nx-layer-${l.id}`} data-layer={l.id}>
            {l.name}
          </Chip>
        ))}
      </div>
    </div>
  );
}

export { MAP_LAYERS };
