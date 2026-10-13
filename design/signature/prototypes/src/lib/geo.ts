export type LngLat = [number, number];
const R = 6371.0088;
const rad = (d: number) => (d * Math.PI) / 180;

export function km(a: LngLat, b: LngLat) {
  const dLat = rad(b[1] - a[1]), dLon = rad(b[0] - a[0]);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a[1])) * Math.cos(rad(b[1])) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
export function fmtKm(d: number) {
  if (d < 1) return `${Math.round(d * 1000 / 50) * 50} m`;
  if (d < 10) return `${d.toFixed(1)} km`;
  return `${Math.round(d)} km`;
}
export function bearingWord(a: LngLat, b: LngLat) {
  const y = Math.sin(rad(b[0] - a[0])) * Math.cos(rad(b[1]));
  const x = Math.cos(rad(a[1])) * Math.sin(rad(b[1])) - Math.sin(rad(a[1])) * Math.cos(rad(b[1])) * Math.cos(rad(b[0] - a[0]));
  const deg = (Math.atan2(y, x) * 180) / Math.PI + 360;
  return ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"][Math.round(deg / 45) % 8];
}
/** A short great-circle polyline (for the provenance threads; at these distances ≈ straight). */
export function arc(a: LngLat, b: LngLat, n = 24): LngLat[] {
  return Array.from({ length: n + 1 }, (_, i) => [a[0] + ((b[0] - a[0]) * i) / n, a[1] + ((b[1] - a[1]) * i) / n] as LngLat);
}
export function dms(lat: number, lon: number) {
  const f = (v: number, p: string, n: string) => {
    const a = Math.abs(v), d = Math.floor(a), m = (a - d) * 60;
    return `${d}°${m.toFixed(2).padStart(5, "0")}′ ${v >= 0 ? p : n}`;
  };
  return `${f(lat, "N", "S")}  ${f(lon, "E", "W")}`;
}
export function latDm(lat: number) {
  const d = Math.floor(lat), m = (lat - d) * 60;
  return `${d}°${m.toFixed(2)}′ N`;
}
