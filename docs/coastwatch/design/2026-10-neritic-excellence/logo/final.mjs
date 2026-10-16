// The three finalists. 48-unit grid; c = palette.
export const FINAL = {
  A: { name: "Shelf", line: "The water over the continental shelf, in section: flat surface, shallow shelf, the break into deep water. One observation in the shallow water.",
    sym: (c) => `<path d="M6 12H42V23.4H35.6C32.4 23.4 30.7 25 29.6 28.2L27.3 34.6C26.2 37.7 24.3 39 21.4 39H6Z" fill="${c.sea}"/>
<circle cx="36.4" cy="17.7" r="2.75" fill="${c.mono ? c.bg : c.dot}"/>`, crop: [6, 12, 36, 27] },
  B: { name: "Depth N", line: "An N drawn as a depth profile: the coast, the shelf sloping gently offshore, the steep slope, and back up. The dot is an observation at the surface over the shelf.",
    sym: (c) => `<defs><clipPath id="bn${c.k ?? ""}"><rect x="0" y="8.6" width="48" height="30.9"/></clipPath></defs>
<path clip-path="url(#bn${c.k ?? ""})" d="M10.6 44V8.6L22.6 12.9C25.9 14.1 27.6 16 28.9 19.2L37.4 39.5V4" stroke="${c.ink}" stroke-width="5.2" stroke-linejoin="miter" stroke-miterlimit="12" fill="none"/>
<circle cx="27.4" cy="7.2" r="3" fill="${c.dot}"/>`, crop: [8, 4.2, 32, 35.3] },
  C: { name: "Soundings", line: "A seafloor profile drawn the way an echo sounder records it: discrete soundings along the track. The shelf, the break, the slope. One sounding is the observation.",
    sym: (c) => { const P = SOUNDINGS; return `<path d="M7 8H39" stroke="${c.sea}" stroke-width="2.6" stroke-linecap="round"/>` + P.map(([x,y],i)=>`<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="2.7" fill="${i===2?(c.mono?c.ink:c.dot):c.ink}"/>`).join("") + (c.mono ? `<circle cx="${P[2][0].toFixed(2)}" cy="${P[2][1].toFixed(2)}" r="1.1" fill="${c.bg}"/>` : ""); }, crop: [4.4, 5.4, 39.4, 37.4] },
};
// equal arc-length points on the profile: shelf (gentle), break, slope
const BZ = [[7,15],[22,17],[31,18.5],[39,40]];
function bez(t){const u=1-t;return [0,1].map(k=>u*u*u*BZ[0][k]+3*u*u*t*BZ[1][k]+3*u*t*t*BZ[2][k]+t*t*t*BZ[3][k]);}
const S=[];let L=0;let prev=bez(0);S.push([0,0]);for(let i=1;i<=400;i++){const p=bez(i/400);L+=Math.hypot(p[0]-prev[0],p[1]-prev[1]);S.push([i/400,L]);prev=p;}
export const SOUNDINGS = Array.from({length:7},(_,j)=>{const target=L*j/6;const s=S.find(([,l])=>l>=target-1e-9)??S[S.length-1];return bez(s[0]);});
export const PAL = {
  light: { k: "l", ink: "#0b2238", sea: "#1d7894", dot: "#e5902a", bg: "#ffffff" },
  dark: { k: "d", ink: "#eef3f8", sea: "#5fc0d8", dot: "#f2ae4a", bg: "#071423" },
  mono: { k: "m", ink: "#0b2238", sea: "#0b2238", dot: "#0b2238", bg: "#ffffff", mono: true },
  monoW: { k: "w", ink: "#ffffff", sea: "#ffffff", dot: "#ffffff", bg: "#0b2238", mono: true },
};
