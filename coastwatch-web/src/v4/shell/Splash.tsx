import { C2, C2_TONES } from "@/v4/brand/c2";

/**
 * The launch screen: the C2 mark drawing itself on the instrument ground, once per browser session.
 *
 * It is plain server HTML and CSS (theme.css, "launch screen"), so it paints and animates from the
 * first frame, before any script, and needs no network asset (it works on an offline reopen). The page
 * renders underneath at the same time. A tiny inline script decides, before the first paint, whether
 * it shows at all: only when a session opens on Explore (where it covers the map loading; a reading
 * page arrives complete in its HTML, and a launch screen there would only hide it), not after the
 * first load of the session, not for automated browsers (unless a test
 * asks for it with `nx-splash-force-v1`), not when `nx-splash-off-v1` is set. SplashGate (client)
 * dismisses it when the app is ready; a CSS animation removes it at 2.5 s even if no script runs.
 */
const DECIDE = `(function(){try{var d=document.documentElement,s=sessionStorage,l=localStorage;if(location.pathname!=="/"||s.getItem("nx-splash-seen-v1")||l.getItem("nx-splash-off-v1")==="1")return;if(navigator.webdriver&&l.getItem("nx-splash-force-v1")!=="1")return;s.setItem("nx-splash-seen-v1","1");d.classList.add("nx-splash-on")}catch(e){}})()`;

const D = C2_TONES.dark;
// the right-hand stub drawn from the tile's edge inwards (the master draws it outwards)
const STUBS = ["M0 26H7.37", "M48 23.2H40.43"];
// two bathymetric contours behind the mark (the start card's family of lines)
const CONTOURS = [0, 1].map((i) => `M-20 ${120 + i * 44} C 80 ${92 + i * 46}, 160 ${170 + i * 36}, 240 ${130 + i * 44} S 380 ${160 + i * 40}, 430 ${140 + i * 44}`);

export function Splash() {
  const size = 64;
  const wordW = ((C2.lockupWidth - 58) / 48) * size;
  return (
    <>
      <script dangerouslySetInnerHTML={{ __html: DECIDE }} />
      <div className="nx-splash" data-testid="splash">
        <p role="status" className="sr-only">
          Loading Neritic
        </p>
        <svg className="nx-splash-sea" viewBox="0 0 400 300" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          {CONTOURS.map((d, i) => (
            <path key={d} d={d} pathLength={1} className="nx-splash-contour" style={{ animationDelay: `${i * 90}ms` }} fill="none" stroke={D.tile} strokeWidth={0.9} />
          ))}
        </svg>
        <div className="nx-splash-mark" aria-hidden="true">
          <svg width={size} height={size} viewBox="0 0 48 48">
            <path className="nx-splash-tile" d={C2.tile} fill={D.tile} />
            <g stroke={D.stub} strokeWidth={C2.stubWidth}>
              {STUBS.map((d) => (
                <path key={d} d={d} pathLength={1} className="nx-splash-stub" />
              ))}
            </g>
            <g className="nx-splash-n">
              <path transform={C2.nAt} d={C2.n} fill={D.n} />
            </g>
          </svg>
          <svg className="nx-splash-word" height={size} width={wordW} viewBox={`58 0 ${C2.lockupWidth - 58} 48`} style={{ marginLeft: (10 / 48) * size }}>
            <path transform={C2.wordAt} d={C2.word} fill={D.word} />
          </svg>
        </div>
      </div>
    </>
  );
}
