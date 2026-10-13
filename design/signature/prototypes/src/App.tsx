import { lazy, Suspense, useEffect, useState } from "react";
import { refreshNetFromUrl } from "./data/net";
import { store } from "./data/store";
import { Index } from "./pages/Index";

/** Hash routes keep every prototype state linkable from a static build: #/route?param=… */
const routes: Record<string, React.LazyExoticComponent<() => React.JSX.Element> | (() => React.JSX.Element)> = {
  "": Index,
  explorer: lazy(() => import("./pages/Explorer").then((m) => ({ default: m.Explorer }))),
  station: lazy(() => import("./pages/Station").then((m) => ({ default: m.Station }))),
  directions: lazy(() => import("./pages/Directions").then((m) => ({ default: m.Directions }))),
  states: lazy(() => import("./pages/States").then((m) => ({ default: m.States }))),
  system: lazy(() => import("./pages/System").then((m) => ({ default: m.System }))),
};

export const route = () => location.hash.replace(/^#\/?/, "").split("?")[0];
export const hashParams = () => new URLSearchParams(location.hash.split("?")[1] ?? "");
export function setHashParams(update: Record<string, string | null>, replace = true) {
  const p = hashParams();
  for (const [k, v] of Object.entries(update)) (v == null ? p.delete(k) : p.set(k, v));
  const q = p.toString();
  const url = `#/${route()}${q ? `?${q}` : ""}`;
  if (replace) history.replaceState(null, "", url); else history.pushState(null, "", url);
}

export function App() {
  const [r, setR] = useState(route());
  useEffect(() => {
    let last = location.hash.split("?")[1] ?? "";
    const on = () => {
      const q = location.hash.split("?")[1] ?? "";
      // a new network profile is a new "session": reload resources under it
      if (new URLSearchParams(q).toString() !== new URLSearchParams(last).toString() && /(net|fail|cache)=/.test(q + last)) {
        refreshNetFromUrl();
        store.reset();
      }
      last = q;
      setR(route());
    };
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  const Page = routes[r] ?? Index;
  return (
    <Suspense fallback={<div style={{ position: "fixed", inset: 0, background: "var(--cw-sea)" }} />}>
      <Page />
    </Suspense>
  );
}
