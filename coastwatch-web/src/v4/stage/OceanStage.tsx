"use client";

import dynamic from "next/dynamic";
import { Component, useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import type { Map as MlMap } from "maplibre-gl";
import { useFocusLink } from "@/v4/motion/focusLink";
import { useInView } from "@/v4/motion/inView";
import { fieldCaveat } from "./caption";
import { stageCanRun } from "./fallback";
import { scrimGradient } from "./scrim";
import { StagePoster } from "./StagePoster";
import type { StageCaption, StageData, StageField, StageMark } from "./types";

// the map's whole weight (MapLibre, the flow layer) is in this chunk, loaded when the stage nears the screen
const StageScene = dynamic(() => import("./StageScene"), { ssr: false, loading: () => null });

const arrivedKey = (key: string) => `nx-stage-arrived:${key}`;
const seen = (key: string) => {
  try {
    return window.sessionStorage.getItem(arrivedKey(key)) === "1";
  } catch {
    return false;
  }
};
const mark = (key: string) => {
  try {
    window.sessionStorage.setItem(arrivedKey(key), "1");
  } catch {
    // storage unavailable: it simply arrives again
  }
};

/** A map that throws (WebGL lost, its chunk unavailable on a saved copy): the poster stays; nothing else changes. */
class SceneBoundary extends Component<{ onFail: () => void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    this.props.onFail();
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

export type OceanStageProps = {
  /** where the camera looks */
  center: [number, number];
  zoom: number;
  /** zoom on a phone-width frame; default zoom − 0.6 */
  phoneZoom?: number;
  /** tilt, 0–50 (the camera looks from the sea toward the coast at about 45) */
  pitch?: number;
  bearing?: number;
  field?: StageField;
  /** measured HF-radar currents as moving streaks, only where there are observed cells */
  currents?: boolean;
  /** what loadStageData gathered on the server */
  data: StageData | null;
  /** the port ringed on the map (its port_code) */
  port?: number | null;
  marks?: StageMark[];
  /** notice and RAMP geometry; features lit through `focus` by properties.id or properties.record_ids; defaults to the official geometry */
  lines?: GeoJSON.FeatureCollection | null;
  /** which mark or line is lit; omitted: the useFocusLink store */
  focus?: string | null;
  /** false by default: no gestures, no scroll capture, the canvas hidden from assistive tech */
  interactive?: boolean;
  /** the legend-as-title; default: the field's own from `data`; false hides it */
  caption?: StageCaption | false;
  /** a key (the port) to ease in once per session; later visits and reduced motion open settled */
  arrival?: string;
  /** a sentence for the screen reader, instead of the generated one */
  alt?: string;
  /** dim and scale as the page scrolls (static without CSS scroll timelines) */
  scrollLinked?: boolean;
  /** the map fills its box; set the box's height in `className`/`style` */
  className?: string;
  style?: CSSProperties;
  testid?: string;
  onPort?: (code: number) => void;
  onPoint?: (p: { lat: number; lon: number }) => void;
  /** the live map, once it exists */
  onMap?: (m: MlMap) => void;
  children?: ReactNode;
};

const KIND: Record<StageCaption["kind"], string> = { observed: "observed", model: "model", outlook: "outlook", analysis: "analysis" };

/** The legend-as-title: what · kind · date, the field's caveat, its ramp. Shown only while the map is drawing that field. */
export function StageCaptionBlock({ caption, field, status, currentsHour }: { caption: StageCaption; field: StageField; status?: "ready" | "failed" | "loading"; currentsHour?: string | null }) {
  const caveat = fieldCaveat(field);
  return (
    <div className="nx-stage-cap" data-testid="stage-caption">
      <p className="nx-stage-cap-t">
        <span>{caption.what}</span>
        <span className="nx-stage-cap-d">{` · ${KIND[caption.kind]}${caption.past ? " (past run)" : ""} · ${caption.date}`}</span>
      </p>
      {caveat ? <p className="nx-stage-cap-c">{caveat}</p> : null}
      {status === "failed" ? <p className="nx-stage-cap-c">Part of this layer could not be loaded. Nothing is drawn in its place.</p> : null}
      {caption.ramp ? (
        <div className="nx-stage-ramp" aria-hidden>
          <i style={{ background: caption.ramp.css }} />
          <span className="nx-stage-ramp-l">
            {caption.ramp.labels.map((l, n) => (
              <span key={n}>{l}</span>
            ))}
          </span>
        </div>
      ) : null}
      {currentsHour ? <p className="nx-stage-cap-c">Streaks: measured currents · {currentsHour}</p> : null}
    </div>
  );
}

/**
 * The ocean stage (design.md §6): a real map as a scene, with the page's own words over it (children, server HTML).
 * The real-isobath poster shows first and stays when the map cannot run. The map loads after first paint when
 * the stage nears the viewport, and fades in over the poster once it has drawn. One per page.
 *
 * Use for: the front of a primary page, where the map is the ground.
 * Do not use for: reading a map closely (that is Explore), or more than once on a page.
 */
export function OceanStage(p: OceanStageProps) {
  const { field = "none", currents = false, interactive = false, arrival, data, caption: captionProp, scrollLinked = false, port = null, children, testid = "ocean-stage" } = p;
  const root = useRef<HTMLDivElement | null>(null);
  const near = useInView(root, { rootMargin: "240px" });
  const visible = useInView(root, { once: false });
  const [can, setCan] = useState<boolean | null>(null);
  const [mode, setMode] = useState<"poster" | "map" | "failed">("poster");
  const [fieldState, setFieldState] = useState<"ready" | "failed" | "loading">("loading");
  const [flowHour, setFlowHour] = useState(false);
  const { focus: storeFocus } = useFocusLink();
  const focus = p.focus === undefined ? storeFocus : p.focus;
  const arriveAtMount = useRef<boolean | null>(null);
  const altId = useId();

  // the first frame paints the poster and the words; only then is the map considered
  useEffect(() => {
    const id = window.setTimeout(() => setCan(stageCanRun()), 0);
    return () => window.clearTimeout(id);
  }, []);
  // read before the arrival is recorded: this mount plays it or does not, whatever happens later
  useLayoutEffect(() => {
    if (arriveAtMount.current == null) arriveAtMount.current = !!arrival && !seen(arrival);
    if (!arrival) return;
    const el = root.current;
    if (el) el.dataset.arrival = arriveAtMount.current && !window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "play" : "settled";
    mark(arrival);
  }, [arrival]);

  const fieldData = field === "none" || !data ? null : data[field];
  const caption = captionProp === false ? null : (captionProp ?? fieldData?.caption ?? null);
  const fallback = can === false || (can === true && data == null);
  const showMap = can === true && near && !!data && mode !== "failed";
  const stageState = mode === "map" ? "map" : fallback ? "poster-only" : "poster";
  const hour = currents && flowHour ? data?.currents?.hour : null;
  const altText = p.alt ?? (caption ? `A map of the water, showing ${caption.what.toLowerCase()} (${KIND[caption.kind]}, ${caption.date}${fieldCaveat(field) ? `, ${fieldCaveat(field)}` : ""}).` : "A map of the water.");

  return (
    <div
      ref={root}
      className={`nx-stage nx-vt-stage ${p.className ?? ""}`}
      style={p.style}
      data-testid={testid}
      data-state={stageState}
      data-arrival={arrival ? "play" : undefined}
      suppressHydrationWarning
    >
      {/* before first paint: a later visit (or reduced motion) opens settled */}
      {arrival ? <script dangerouslySetInnerHTML={{ __html: arrivalScript(arrivedKey(arrival)) }} /> : null}
      <div className={`nx-stage-scene ${scrollLinked ? "nx-scroll-stage" : ""}`}>
        <StagePoster />
        <div className="nx-stage-map" data-interactive={interactive ? "true" : "false"} data-on={mode === "map"} aria-hidden={interactive ? undefined : true}>
          {showMap && data ? (
            <SceneBoundary onFail={() => setMode("failed")}>
              <StageScene
                {...p}
                field={field}
                currents={currents}
                data={data}
                focus={focus ?? null}
                port={port}
                interactive={interactive}
                visible={visible}
                arrive={!!arriveAtMount.current}
                onShown={() => setMode("map")}
                onFail={() => setMode("failed")}
                onFieldState={setFieldState}
                onFlow={setFlowHour}
              />
            </SceneBoundary>
          ) : null}
        </div>
        <div className="nx-stage-scrim" style={{ background: scrimGradient() }} aria-hidden />
      </div>
      <p id={altId} className="sr-only" data-testid="stage-alt">
        {altText}
      </p>
      {mode === "map" && caption ? (
        <div className="nx-stage-cap-at">
          <StageCaptionBlock caption={caption} field={field} status={fieldState} currentsHour={hour} />
        </div>
      ) : null}
      {stageState === "poster-only" || mode === "failed" ? (
        <p className="nx-stage-note" data-testid="stage-poster-note">
          The live map is not loading here. These lines are the Monterey Canyon chart, not your waters.
        </p>
      ) : null}
      {children != null ? <div className="nx-stage-over">{children}</div> : null}
    </div>
  );
}

function arrivalScript(key: string) {
  return `(function(){try{var e=document.currentScript.parentElement;if(sessionStorage.getItem(${JSON.stringify(key)})==="1"||matchMedia("(prefers-reduced-motion: reduce)").matches)e.dataset.arrival="settled"}catch(_){}})()`;
}
