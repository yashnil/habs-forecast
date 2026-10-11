/**
 * A small line of daily values (M5 inspector summary). Days without a value break the line;
 * nothing is drawn across a gap. The full chart, with axes and values, opens below it.
 */
export function Sparkline({ points, domain, label, width = 76, height = 24 }: { points: { date: string; value: number | null }[]; domain: [number, number]; label: string; width?: number; height?: number }) {
  const pts = points.filter((p) => p.date);
  if (pts.length < 2) return null;
  const t = pts.map((p) => Date.parse(`${p.date}T00:00:00Z`));
  const t0 = Math.min(...t);
  const t1 = Math.max(...t);
  const x = (i: number) => (t1 === t0 ? 0 : ((t[i] - t0) / (t1 - t0)) * (width - 2) + 1);
  const y = (v: number) => height - 2 - ((Math.min(domain[1], Math.max(domain[0], v)) - domain[0]) / (domain[1] - domain[0])) * (height - 4);
  const runs: string[] = [];
  let cur: string[] = [];
  pts.forEach((p, i) => {
    const gap = i > 0 && t[i] - t[i - 1] > 86_400_000 * 1.5;
    if (p.value == null || gap) {
      if (cur.length > 1) runs.push(cur.join(" "));
      cur = [];
    }
    if (p.value != null) cur.push(`${x(i).toFixed(1)},${y(p.value).toFixed(1)}`);
  });
  if (cur.length > 1) runs.push(cur.join(" "));
  const last = [...pts].reverse().find((p) => p.value != null);
  const li = last ? pts.indexOf(last) : -1;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label} data-testid="sparkline" className="shrink-0">
      <line x1={0} x2={width} y1={height - 2} y2={height - 2} stroke="var(--cw-hairline-strong, #c8d1db)" strokeWidth={1} />
      {runs.map((r) => (
        <polyline key={r} points={r} fill="none" stroke="var(--cw-forecast)" strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
      ))}
      {last && li >= 0 && <circle cx={x(li)} cy={y(last.value!)} r={2} fill="var(--cw-forecast)" />}
    </svg>
  );
}
