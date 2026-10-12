# Map label glyphs (self-hosted)

Signed-distance-field glyph ranges for the Ocean Map and Bloom station map labels, so place names and
port markers never wait on a third-party font server. Served from `/fonts/{fontstack}/{range}.pbf`;
`lib/basemap.ts` (`transformRequest`) sends any range not listed here to OpenFreeMap as before.

- Fonts: Noto Sans Regular and Noto Sans Italic (Google / Noto project), SIL Open Font License 1.1.
- Glyph ranges as served by OpenFreeMap (`https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf`,
  built from the OpenMapTiles fonts repository), recorded 2026-10-11.
- Ranges: 0-255, 256-511, 512-767, 768-1023, 7680-7935, 8192-8447, 8448-8703, 8704-8959 (Latin,
  Latin extended, combining marks, general punctuation, letterlike symbols, mathematical operators).
