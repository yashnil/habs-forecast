// Shared helpers for board.html and summary.html (Neritic final comparison).
const SVG = '../round2/svg/', ICO = '../round2/icons/', SH = 'shots/';
const NAME = { r2: 'R2 · Bight', c2: 'C2 · Chart label' };

// an enlarged screenshot: data-scale multiplies the natural pixel size (nearest-neighbour when > 1)
const shot = (n, scale = 1, extra = '') => `<img src="${SH}${n}.png" data-scale="${scale}" alt="" ${extra}>`;
// a window onto a screenshot: crop (x, y, w, h) in CSS px of the 1x layout; dsf = which render; scale = display factor
function crop(n, { x = 0, y = 0, w, h, dsf = 1, scale = 1, glare = 0 }) {
  const f = scale; // display px per CSS px of the 1x layout = dsf render shown at (scale/dsf)
  const src = `${SH}${n}${dsf === 2 ? '@2x' : ''}.png`;
  return `<div class="crop" style="width:${w * f}px;height:${h * f}px">
    <img src="${src}" data-fit="${f / dsf}" style="left:${-x * f}px;top:${-y * f}px;image-rendering:${f / dsf > 1 ? 'pixelated' : 'auto'}" alt="">
    ${glare ? `<div class="veil" style="opacity:${glare}"></div>` : ''}</div>`;
}
function fit() {
  for (const im of document.querySelectorAll('img[data-scale], img[data-fit]')) {
    const go = () => {
      const s = +(im.dataset.scale || im.dataset.fit);
      im.style.width = im.naturalWidth * s + 'px'; im.style.height = im.naturalHeight * s + 'px';
      if (im.dataset.scale) im.style.imageRendering = s > 1 ? 'pixelated' : 'auto';
    };
    im.complete && im.naturalWidth ? go() : im.addEventListener('load', go);
  }
}
const appIcon = (k, size, shape) => {
  // shape: square | ios (rounded 22.37 %) | circle (full bleed) | safe (maskable: only the centre 80 % circle survives)
  const r = shape === 'square' ? 0 : shape === 'ios' ? size * 0.2237 : size / 2;
  const inner = shape === 'safe' ? size / 0.8 : size;
  return `<div class="app" style="width:${size}px;height:${size}px;border-radius:${r}px"><img src="${SVG}${k}-app.svg" style="width:${inner}px;height:${inner}px;margin:${(size - inner) / 2}px" alt=""></div>`;
};
