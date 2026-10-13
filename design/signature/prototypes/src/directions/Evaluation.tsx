/**
 * Honest scores from the built prototypes (1 = poor, 5 = strong; implementation cost: 5 = cheapest).
 * Not stakeholder-validated: these are a designer's judgement against the brief, recorded so the
 * recommendation can be argued with.
 */
const CRITERIA = ["Usability", "Clarity", "Distinctiveness", "Scientific accuracy", "Implementation cost"] as const;
type Row = { d: string; scores: number[]; why: string[] };
const ROWS: Row[] = [
  {
    d: "A · Marine Observatory",
    scores: [3, 4, 4, 5, 4],
    why: [
      "Fast for repeat and research use; the dark rail is dense for a first-time harvester.",
      "Values, units and dates read instantly; class lines keep the coast visible.",
      "A real graticule, exact cells and class lines look like an instrument, not a dashboard.",
      "Every mark is exact: cell edges, class boundaries, the limit ruler, dated samples.",
      "Closest to today's dark map and tokens; graticule is ~60 lines.",
    ],
  },
  {
    d: "B · Living Coast",
    scores: [4, 4, 3, 3, 3],
    why: [
      "Sentences answer the visitor's question with almost no learning curve.",
      "Calm and legible, but values hide in prose; harder to compare dates and numbers.",
      "Pleasant but closest to generic light map apps.",
      "The lens hides forecast cells outside 14 km; we must say \"hidden, not absent\" every time — a risk.",
      "A second (light) basemap palette and light variants of every map colour to maintain.",
    ],
  },
  {
    d: "C · Editorial Atlas",
    scores: [3, 5, 5, 4, 3],
    why: [
      "Excellent for reading and sharing; the plate is less fluid to pan on a phone.",
      "The headline states the answer; numbered notes link each claim to its place on the map.",
      "Unlike any HAB or ocean data site: a plate with caption, dateline and margin notes.",
      "Numbered callouts make provenance explicit; low-contrast veil can undersell high classes.",
      "Paper surfaces exist already; the plate map needs its own light style and callout layer.",
    ],
  },
];

export function Evaluation() {
  return (
    <details className="dir-eval">
      <summary>Evaluation</summary>
      <div className="dir-eval-body">
        <table>
          <caption>Scores from the built views, 1–5 (cost: 5 = cheapest). Designer judgement, not user-tested.</caption>
          <thead><tr><th scope="col">Direction</th>{CRITERIA.map((c) => <th key={c} scope="col">{c}</th>)}<th scope="col">Total</th></tr></thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r.d}>
                <th scope="row">{r.d}</th>
                {r.scores.map((s, i) => <td key={i} title={r.why[i]}><span className="num">{s}</span><span className="dir-why">{r.why[i]}</span></td>)}
                <td className="num"><b>{r.scores.reduce((a, b) => a + b, 0)}</b></td>
              </tr>
            ))}
          </tbody>
        </table>
        <h2>Recommendation: “Atlas reading, Observatory instrument”</h2>
        <p>Use C's structure for every reading surface — paper, a serif headline that states the answer, a dateline, and numbered notes that point at the map — because that is what makes CoastWatch's provenance visible and memorable. Keep A's map as the instrument: dark sea, exact cells, class lines or a lens instead of a loud filled field, a graticule in Research mode. Take B's restraint, not its look: generous spacing and one-time, purposeful motion in Coastal mode; do not adopt its light basemap or a lens that hides cells without hatching.</p>
        <p>Why not B alone: it is the easiest to read, but its lens hides model cells outside the circle with nothing in their place, and plain-sentence numbers are hard to compare. Why not A alone: the dark rail is efficient but intimidating for a first visit and weak for sharing.</p>
      </div>
    </details>
  );
}
