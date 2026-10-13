/**
 * Purpose onboarding where the *map answers the choice*. Not a modal: a small card over the map.
 * Choosing a purpose sets defaults (never facts) and immediately frames the map for it, with one
 * line that says what changed and that it can be changed. "Just exploring" starts a three-stop
 * guided journey along the coast. Skippable; remembered on this device.
 */
export type Purpose = "commercial" | "shellfish" | "research" | "explore";
export const PURPOSES: { id: Purpose; label: string; sub: string; caption: string }[] = [
  { id: "commercial", label: "Commercial fishing", sub: "Offshore waters, forecast close to you", caption: "Showing offshore waters with NOAA's forecast around the point you choose, and the notices for your species." },
  { id: "shellfish", label: "Shellfish or recreational harvest", sub: "The shoreline and official notices", caption: "Showing the shoreline, official notice limits and the nearest shore samples. Forecast off by default — it does not cover most nearshore water." },
  { id: "research", label: "Research and science", sub: "Every layer, exact values", caption: "Research presentation: exact cell values, the model's cell grid and data sources, statewide." },
  { id: "explore", label: "Just exploring", sub: "A three-stop tour along the coast", caption: "" },
];

export function PurposeCard({ onChoose, onSkip }: { onChoose: (p: Purpose) => void; onSkip: () => void }) {
  return (
    <section className="purpose enter" aria-labelledby="purpose-h">
      <h2 id="purpose-h" className="t-headline">What brings you to the coast?</h2>
      <p className="t-meta">Optional. It sets what the map shows first, never what is true.</p>
      <ul className="purpose-list">
        {PURPOSES.map((p) => (
          <li key={p.id}>
            <button className="purpose-btn" onClick={() => onChoose(p.id)}>
              <span className="t-ui">{p.label}</span>
              <span className="t-meta">{p.sub}</span>
              <span className="purpose-arrow" aria-hidden="true">→</span>
            </button>
          </li>
        ))}
      </ul>
      <button className="btn btn-quiet purpose-skip" onClick={onSkip}>Skip</button>
    </section>
  );
}

export const TOUR = [
  { at: [-122.03, 36.82] as [number, number], zoom: 9.2, show: "base" as const, title: "Monterey Bay", text: "Amber lines are official limits. Two notices on northern anchovy end at Point Lobos, 36°31.46′ N — the line south of Monterey." },
  { at: [-122.017, 36.958] as [number, number], zoom: 11, show: "base" as const, title: "Santa Cruz Wharf", text: "A green dot is a place where people sample the water, about weekly. A filled dot was sampled in the last two weeks; a ring, longer ago." },
  { at: [-122.4, 36.9] as [number, number], zoom: 8.4, show: "forecast" as const, title: "NOAA's forecast", text: "Violet cells are a model's probability, one 3 km cell at a time. Hatching is where it has no value — often right along the shore." },
];
