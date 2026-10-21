import { MARK_INDEX_DEPTH, MARK_TILE, MARK_TONES, MARK_WORD, strokeOf } from "@/v4/brand/mark";
import { MARK_FIELD, MARK_HERO } from "@/v4/brand/mark.geometry";

/**
 * The launch screen: Monterey Canyon draws itself, the mark resolves out of it, the wordmark follows,
 * and the whole thing dives into the map. Once per device: the first visit ever, and only when that
 * visit opens on Explore.
 *
 * The drawing is the mark's own data (src/v4/brand/mark.geometry.ts, real isobaths): the five lines
 * of the symbol draw outward from the canyon head at full size, exactly over the same isobaths in a
 * faint field of the bay around them; the tile fills in behind them; the symbol then shrinks into
 * the lockup while the field drifts more slowly behind it (parallax), and the wordmark wipes in.
 *
 * It is plain server HTML and CSS (theme.css, "launch screen"): it paints and animates from the first
 * frame, before any script, with no network asset. The page renders underneath at the same time. A tiny
 * inline script decides, before the first paint, whether it shows at all: only on this device's first
 * visit (a localStorage flag, `nx-splash-seen-v2`, set on that visit whatever page it opens; a device
 * that already keeps Neritic's settings is a returning reader too), only when that visit opens on Explore
 * (where it covers the map loading; never on a reading page, whose content is in its first HTML), never
 * on a saved copy reopened from a field pack (the service worker's mark, set before this script runs),
 * not for automated browsers unless a test asks with `nx-splash-force-v1`, not when `nx-splash-off-v1`
 * is set. A tap, click or key skips it at once, even before the app's script has loaded. SplashGate
 * (MotionRoot) holds that one first showing at least 1.8 s and steps it aside when the map is ready; a
 * CSS animation removes it by 3 s even if no script runs. A returning reader is never held.
 */
const EXIT_MS = 450;
const SEEN = "nx-splash-seen-v2";
const DECIDE = `(function(){try{var d=document.documentElement,l=localStorage;if(l.getItem("nx-splash-off-v1")==="1"||self.__NERITIC_SAVED__||l.getItem("${SEEN}"))return;var force=l.getItem("nx-splash-force-v1")==="1";if(navigator.webdriver&&!force)return;var back=false;if(!force)for(var i=0;i<l.length;i++){var k=l.key(i)||"";if(/^(nx-|cw-|neritic-)/.test(k)&&k.indexOf("nx-splash-")!==0){back=true;break}}l.setItem("${SEEN}","1");if(back||location.pathname!=="/")return;d.classList.add("nx-splash-on");var skip=function(e){if(!d.classList.contains("nx-splash-on")||d.classList.contains("nx-splash-out"))return;if(e&&e.type==="keydown"&&(e.key==="Tab"||e.key==="Shift"))return;d.classList.add("nx-splash-out");setTimeout(function(){d.classList.remove("nx-splash-on","nx-splash-out")},${EXIT_MS});removeEventListener("pointerdown",skip,true);removeEventListener("keydown",skip,true)};addEventListener("pointerdown",skip,true);addEventListener("keydown",skip,true)}catch(e){}})()`;

const D = MARK_TONES.dark;
// the field's draw order: from the 200 m isobath outward, shallower and deeper alike
const fieldDepths = [...new Set(MARK_FIELD.lines.map((l) => l.depth))].sort((a, b) => Math.abs(a - MARK_INDEX_DEPTH) - Math.abs(b - MARK_INDEX_DEPTH) || a - b);
const heroDepths = [...new Set(MARK_HERO.map((l) => l.depth))].sort((a, b) => Math.abs(a - MARK_INDEX_DEPTH) - Math.abs(b - MARK_INDEX_DEPTH) || a - b);
const H = MARK_FIELD.half;

export function Splash() {
  return (
    <>
      <script dangerouslySetInnerHTML={{ __html: DECIDE }} />
      <div className="nx-splash" data-testid="splash" style={{ "--nx-wr": MARK_WORD.width / 48 } as React.CSSProperties}>
        <p role="status" className="sr-only">
          Loading Neritic
        </p>
        <div className="nx-splash-depth" aria-hidden="true" />
        <div className="nx-splash-drift" aria-hidden="true">
          <svg className="nx-splash-field" viewBox={`${24 - H} ${24 - H} ${2 * H} ${2 * H}`}>
            <g fill="none" strokeLinecap="round">
              {MARK_FIELD.lines.map((l, i) => (
                <path
                  key={i}
                  d={l.d}
                  pathLength={1}
                  className={l.depth === MARK_INDEX_DEPTH ? "nx-splash-iso nx-splash-iso-index" : l.depth === 0 ? "nx-splash-iso nx-splash-iso-coast" : "nx-splash-iso"}
                  style={{ "--i": fieldDepths.indexOf(l.depth) } as React.CSSProperties}
                />
              ))}
            </g>
          </svg>
        </div>
        <div className="nx-splash-stage" aria-hidden="true">
        <div className="nx-splash-hero">
          <svg viewBox="0 0 48 48" width="100%" height="100%">
            <defs>
              <clipPath id="nx-splash-clip">
                <path d={MARK_TILE} />
              </clipPath>
            </defs>
            <path className="nx-splash-tile" d={MARK_TILE} fill={D.tile} />
            <g clipPath="url(#nx-splash-clip)" fill="none" strokeLinecap="round">
              {MARK_HERO.map((l, i) => (
                <path
                  key={i}
                  d={l.d}
                  pathLength={1}
                  className="nx-splash-line"
                  stroke={l.depth === MARK_INDEX_DEPTH ? D.index : D.line}
                  strokeWidth={strokeOf("full", l.depth)}
                  style={{ "--i": heroDepths.indexOf(l.depth) } as React.CSSProperties}
                />
              ))}
            </g>
          </svg>
        </div>
        <svg className="nx-splash-word" viewBox={`-1 0 ${MARK_WORD.width + 2} 48`}>
          <path transform={`translate(0 ${MARK_WORD.baseline})`} d={MARK_WORD.d} fill={D.word} />
        </svg>
        </div>
        <p className="nx-splash-skip" aria-hidden="true">
          <span className="nx-splash-skip-touch">Tap to skip</span>
          <span className="nx-splash-skip-mouse">Click or press any key to skip</span>
        </p>
      </div>
    </>
  );
}
