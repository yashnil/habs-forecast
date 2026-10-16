// Symbol candidates on a 48-unit grid. `c` = palette {ink, sea, dot, bg}
export const SYM = {
  A1: (c) => `<path d="M9 12H39" stroke="${c.sea}" stroke-width="4.5" stroke-linecap="round" fill="none"/>
<path d="M9 23.5L25.5 27.6C29.2 28.5 31.2 30.6 32.6 34L35.8 40.5" stroke="${c.ink}" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
<circle cx="17" cy="18.2" r="3.1" fill="${c.dot}"/>`,
  A2: (c) => `<path d="M8 13H40V22.5L33.5 22.5C31 22.5 29.6 24 28.6 26.6L24.6 37.5C24 39 22.8 40 21.2 40H8Z" fill="${c.sea}" fill-opacity="${c.mono ? 1 : 1}"/>
<circle cx="${c.mono ? 33.6 : 33.6}" cy="17.6" r="2.9" fill="${c.mono ? c.bg ?? '#fff' : c.dot}"/>`,
  B1: (c) => `<path d="M11 39V9.5L23 14.5C26.8 16.1 28.6 18.4 30 22.5L37 39V9" stroke="${c.ink}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
<circle cx="27.2" cy="8.6" r="3" fill="${c.dot}"/>`,
  B2: (c) => `<path d="M10 39V9L22.5 13.2C25.8 14.3 27.5 16.2 28.8 19.4L37.5 39V9" stroke="${c.ink}" stroke-width="5.2" stroke-linecap="butt" stroke-linejoin="round" fill="none"/>
<circle cx="27.6" cy="8.4" r="3" fill="${c.dot}"/>`,
  C2: (c) => `<path d="M34.5 5C34 12.5 30 18.6 25.2 24C22.8 26.8 22.8 29.4 25.4 30.9C28.6 32.8 35.6 33 43.5 34" stroke="${c.ink}" stroke-width="4.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
<path d="M27.6 5C26.8 11.6 23 16.8 18.6 21.6C14.8 25.8 15 32.4 19.2 36C23.6 39.6 32.4 40.4 43.5 41.2L43.5 44.6C31.6 43.8 21.6 43 16.2 38.6C10.6 34 10.2 25.4 15 19.6C18.8 15 22.2 10.8 22.9 5Z" fill="${c.sea}"/>
<circle cx="${c.mono ? 21.6 : 21.6}" cy="13.4" r="2.8" fill="${c.dot}"/>`,
  C3: (c) => `<rect x="5" y="5" width="38" height="38" rx="3" fill="none" stroke="${c.ink}" stroke-width="2.4"/>
<path d="M33 5.5C32.6 12 29 17.2 24.8 22C22.6 24.6 22.6 27 25 28.4C28 30.2 34.6 30.6 42.5 31.4" stroke="${c.ink}" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
<path d="M26 5.5C25.2 11.2 21.8 15.6 17.8 20C14.6 23.6 14.8 29.4 18.4 32.6C22.2 35.8 30 36.8 42.5 37.6" stroke="${c.sea}" stroke-width="2.2" fill="none"/>
<path d="M19.4 5.5C18.6 10 15.8 13.4 12.6 17C9 21 9.2 30.4 14 35.4C18.8 40.4 30 42 42.5 42.6" stroke="${c.sea}" stroke-width="1.6" stroke-opacity="${c.mono ? 1 : .55}" fill="none"/>
<circle cx="21.6" cy="15.4" r="2.6" fill="${c.dot}"/>`,
  C4: (c) => { const P = [[7,17],[13,18],[19,19.2],[25,20.6],[30.6,24.2],[34.8,30.6],[38.6,38]]; return P.map(([x,y],i)=>`<circle cx="${x}" cy="${y}" r="${i===2?3.4:2.9}" fill="${i===2?c.dot:c.ink}"/>`).join("") + `<path d="M7 9.5H41" stroke="${c.sea}" stroke-width="2.6" stroke-linecap="round"/>`; },
  C5: (c) => { const P = [[7,15],[12.5,16],[18,17.2],[23.5,18.6],[28.6,21.6],[32.6,27.4],[36,34.6],[39.2,42]]; return P.map(([x,y],i)=>`<circle cx="${x}" cy="${y}" r="${i===2?3.3:2.6}" fill="${i===2?c.dot:c.ink}"/>`).join(""); },
};
export const PAL = {
  color: { ink: "#0b2238", sea: "#1f7e9a", dot: "#e8962e", k: "c" },
  dark: { ink: "#eef3f8", sea: "#6cc6dc", dot: "#f2ae4a", k: "d" },
  mono: { ink: "#0b2238", sea: "#0b2238", dot: "#0b2238", bg: "#ffffff", mono: true, k: "m" },
  monoW: { ink: "#ffffff", sea: "#ffffff", dot: "#ffffff", bg: "#0b2238", mono: true, k: "w" },
};
export const svg = (body, s, vb = "0 0 48 48") => `<svg xmlns="http://www.w3.org/2000/svg" width="${s}" height="${s}" viewBox="${vb}">${body}</svg>`;
