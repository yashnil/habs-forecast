import { useEffect, useState } from "react";
import { hashParams, setHashParams } from "../App";
import { Atlas } from "../directions/Atlas";
import { Evaluation } from "../directions/Evaluation";
import { LivingCoast } from "../directions/LivingCoast";
import { Observatory } from "../directions/Observatory";
import { useDirectionState } from "../directions/shared";
import "./directions.css";

/**
 * Phase 2 — three identities rendering the same real state (#/directions?d=a|b|c). `frame=mobile`
 * renders only the direction, for the 390 px phone frame (an iframe on desktop).
 */
type D = "a" | "b" | "c";
const NAMES: Record<D, { name: string; line: string }> = {
  a: { name: "Marine Observatory", line: "A precise instrument" },
  b: { name: "Living Coast", line: "Calm, light, geographic" },
  c: { name: "Editorial Atlas", line: "Map as plate, text that answers" },
};

export function Directions() {
  const p = hashParams();
  const [d, setD] = useState<D>(((p.get("d") as D) ?? "a"));
  const frame = p.get("frame") === "mobile";
  const [phone, setPhone] = useState(p.get("phone") === "1");
  const st = useDirectionState();
  useEffect(() => { if (!frame) setHashParams({ d, phone: phone ? "1" : null }); }, [d, phone, frame]);
  const mobile = frame || (typeof innerWidth !== "undefined" && innerWidth < 761);
  const view = !st.ready ? <div className="dir-loading">Loading the shared state…</div>
    : d === "a" ? <Observatory key="a" st={st} mobile={mobile} /> : d === "b" ? <LivingCoast key="b" st={st} mobile={mobile} /> : <Atlas key="c" st={st} mobile={mobile} />;
  if (frame) return <div className="dir-root dir-frame">{view}</div>;
  const order: D[] = ["a", "b", "c"];
  return (
    <div className="dir-root">
      <div className="dir-switch" role="radiogroup" aria-label="Visual direction">
        {order.map((k) => (
          <button key={k} role="radio" aria-checked={d === k} tabIndex={d === k ? 0 : -1} onClick={() => setD(k)}
            onKeyDown={(e) => {
              const s = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
              if (!s) return;
              e.preventDefault();
              const n = order[(order.indexOf(d) + s + 3) % 3];
              setD(n);
              (e.currentTarget.parentElement?.querySelector(`[data-k="${n}"]`) as HTMLElement | null)?.focus();
            }} data-k={k}>
            <span className="dir-switch-k">{k.toUpperCase()}</span>
            <span className="dir-switch-n">{NAMES[k].name}</span>
          </button>
        ))}
        <button className="dir-phone" aria-pressed={phone} onClick={() => setPhone(!phone)}>Phone view</button>
        <Evaluation />
      </div>
      <div className="dir-view">{view}</div>
      {phone && (
        <div className="dir-phoneframe" aria-label={`${NAMES[d].name} at 390 × 844`}>
          <iframe key={d} title={`${NAMES[d].name}, phone`} src={`#/directions?d=${d}&frame=mobile`} width={390} height={844} />
        </div>
      )}
    </div>
  );
}
