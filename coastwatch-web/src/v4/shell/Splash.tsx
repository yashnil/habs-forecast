import { MARK_INDEX_DEPTH, MARK_TILE, MARK_TONES, MARK_WORD, strokeOf } from "@/v4/brand/mark";
import { MARK_FIELD, MARK_HERO } from "@/v4/brand/mark.geometry";

/**
 * The launch screen, "Sounding" (v9, docs/coastwatch/v9/motion.md §3): depth soundings appear as points
 * along the real Monterey Canyon isobaths and join into contours; the canyon head's own contours grow into
 * the mark and the lockup forms; then the chart recedes, the ground clears to the live map, and the lockup
 * settles into the header's brand. About 3.5 s (`--nx-o` in theme.css stretches the whole play; 1 is the
 * designed tempo).
 *
 * Every line is the mark's own data (src/v4/brand/mark.geometry.ts, real isobaths of the NOAA Coastal
 * Relief Model): MARK_FIELD for the bay, MARK_HERO for the symbol (drawn over the field's 100-300 m lines).
 *
 * It is plain server HTML and CSS (theme.css, "launch screen"): the soundings, contours, mark and lockup
 * paint and animate from the first frame, before any script, with no network asset. The handoff waits for
 * the map, so SplashGate (MotionRoot) runs it once the map is ready; without a script, a CSS cap fades the
 * screen out. A tiny inline script decides, before the first paint, whether it
 * shows at all: only on this device's first visit (a localStorage flag, `nx-splash-seen-v2`, set on that
 * visit whatever page it opens; a device that already keeps Neritic's settings is a returning reader too),
 * only when that visit opens on Explore (never on a reading page, whose content is in its first HTML),
 * never on a saved copy reopened from a field pack (the service worker's mark, set before this script
 * runs), not for automated browsers unless a test asks with `nx-splash-force-v1`, not when
 * `nx-splash-off-v1` is set. A tap, click or key skips it at once (a 200 ms fade), even before the app's
 * script has loaded. It sits under the header, so the workspaces, search and notices stay usable; only the
 * header's brand waits for the mark to land in it.
 */
const EXIT_MS = 200;
const SEEN = "nx-splash-seen-v2";
const DECIDE = `(function(){try{var d=document.documentElement,l=localStorage;if(l.getItem("nx-splash-off-v1")==="1"||self.__NERITIC_SAVED__||l.getItem("${SEEN}"))return;var force=l.getItem("nx-splash-force-v1")==="1";if(navigator.webdriver&&!force)return;var back=false;if(!force)for(var i=0;i<l.length;i++){var k=l.key(i)||"";if(/^(nx-|cw-|neritic-)/.test(k)&&k.indexOf("nx-splash-")!==0){back=true;break}}l.setItem("${SEEN}","1");if(back||location.pathname!=="/")return;d.classList.add("nx-splash-on");var skip=function(e){if(!d.classList.contains("nx-splash-on")||d.classList.contains("nx-splash-out"))return;if(e&&e.type==="keydown"&&(e.key==="Tab"||e.key==="Shift"))return;d.classList.add("nx-splash-out");setTimeout(function(){d.classList.remove("nx-splash-on","nx-splash-out","nx-splash-land")},${EXIT_MS});removeEventListener("pointerdown",skip,true);removeEventListener("keydown",skip,true)};addEventListener("pointerdown",skip,true);addEventListener("keydown",skip,true)}catch(e){}})()`;

const D = MARK_TONES.dark;
// the field's order: from the 200 m isobath outward, shallower and deeper alike
const fieldDepths = [...new Set(MARK_FIELD.lines.map((l) => l.depth))].sort((a, b) => Math.abs(a - MARK_INDEX_DEPTH) - Math.abs(b - MARK_INDEX_DEPTH) || a - b);
const H = MARK_FIELD.half;

/** A field isobath's opacity by depth: the coast and the 200 m index carry the chart; the deep falls away. */
const fieldOpacity = (depth: number) =>
  depth === 0 ? 0.45 : depth === MARK_INDEX_DEPTH ? 0.7 : depth <= 150 ? 0.32 : depth <= 600 ? 0.26 : Math.max(0.12, 0.22 - ((depth - 800) / 1200) * 0.1);

/**
 * The decision, on every page (root layout): a few hundred bytes that mark the device's first visit
 * whatever page it opens, and switch the launch screen on only when that visit opens on Explore.
 */
export function SplashDecide() {
  return <script dangerouslySetInnerHTML={{ __html: DECIDE }} />;
}

/** The launch screen itself: only in Explore's HTML, so reading pages carry none of its drawing. */
export function Splash() {
  const wr = { "--nx-wr": MARK_WORD.width / 48 } as React.CSSProperties;
  return (
    <>
    <div className="nx-splash" data-testid="splash" style={wr}>
      <p role="status" className="sr-only">
        Loading Neritic. Press any key to skip.
      </p>
      {/* the bay's chart: soundings that join into isobaths; the handoff lands it on the map's Monterey Canyon */}
      <div className="nx-splash-chart" aria-hidden="true" data-testid="splash-chart">
        <svg className="nx-splash-field" viewBox={`${24 - H} ${24 - H} ${2 * H} ${2 * H}`}>
          <g fill="none" strokeLinecap="round">
            {MARK_FIELD.lines.map((l, i) => (
              <path
                key={i}
                d={l.d}
                className={l.depth === MARK_INDEX_DEPTH ? "nx-splash-iso nx-splash-iso-index" : l.depth === 0 ? "nx-splash-iso nx-splash-iso-coast" : "nx-splash-iso"}
                style={{ "--i": fieldDepths.indexOf(l.depth), "--o": fieldOpacity(l.depth) } as React.CSSProperties}
              />
            ))}
          </g>
        </svg>
      </div>
    </div>
    {/* the lockup on its own layer, above the header (it takes no clicks), so it can land on the header's brand */}
    <div className="nx-splash nx-splash-top" aria-hidden="true" style={wr}>
      <div className="nx-splash-stage" data-testid="splash-lockup">
        <div className="nx-splash-hero">
          <svg viewBox="0 0 48 48" width="100%" height="100%">
            <defs>
              <clipPath id="nx-splash-clip">
                <path d={MARK_TILE} />
              </clipPath>
            </defs>
            <path className="nx-splash-tile" d={MARK_TILE} fill={D.tile} />
            {/* the canyon head's contours over the field's own: they thicken into the mark, never redraw */}
            <g className="nx-splash-lines" clipPath="url(#nx-splash-clip)" fill="none" strokeLinecap="round">
              {MARK_HERO.map((l, i) => (
                <path
                  key={i}
                  d={l.d}
                  className={l.depth === MARK_INDEX_DEPTH ? "nx-splash-line nx-splash-line-index" : "nx-splash-line"}
                  data-depth={l.depth}
                  stroke={l.depth === MARK_INDEX_DEPTH ? D.index : D.line}
                  style={{ "--w": strokeOf("full", l.depth) } as React.CSSProperties}
                />
              ))}
            </g>
          </svg>
        </div>
        <svg className="nx-splash-word" viewBox={`-1 0 ${MARK_WORD.width + 2} 48`}>
          <path transform={`translate(0 ${MARK_WORD.baseline})`} d={MARK_WORD.d} fill={D.word} />
        </svg>
      </div>
    </div>
    </>
  );
}
