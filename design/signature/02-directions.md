# 02 — Three directions and the recommendation

Live: `#/directions?d=a|b|c` (with a phone view and an evaluation panel).
Images: [desktop](img/directions-desktop.jpg) · [mobile](img/directions-mobile.jpg).

All three render the **same computed state** from the published snapshot:
- C-HARM particulate domoic acid for Mon Oct 12 (lead 2, issued Oct 11, "Not confirmed" because the last fetch failed);
- the water point off Santa Cruz (cell value 0.6946, class 60–70 %);
- Santa Cruz Wharf, last sampled Sep 30 (0.20 ng/mL pDA);
- the three notices covering the point, with the anchovy notices' 36°31.46′ N south limit 31 km away;
- the SN26-018 discrepancy fixture;
- the gaps.

## A — Marine Observatory

A precise instrument:
- dark full-bleed map;
- a slim dark inspector rail with large tabular values;
- class lines along exact cell edges;
- a live graticule on the map edge;
- an instrument bar showing the "for" and "issued" dates.

**Strengths:** best scientific legibility, and the exact cells are unmistakable. Cheapest to build, because it is the current dark map plus a rail.
**Weaknesses:** reading long official text on a dark rail is tiring. Coastal users get an instrument when they want an answer.

## B — Living Coast

Calm and light:
- a re-tinted light sea and sand basemap;
- the forecast only inside a lens around the point;
- a large serif place name;
- the ledger written as plain sentences;
- one selection ripple and no other motion.

**Strengths:** the most approachable; the sentences read naturally.
**Weaknesses:**
- **The lens hides cells outside the circle with nothing in their place.** It needs a "hidden, not absent" caption every time, which is a scientific-clarity cost.
- The light sea loses the bathymetric drama that is CoastWatch's best free asset.
- Light basemap colours have little room for the violet probability palette.

## C — Editorial Atlas

Paper first, with the map as a figure:
- a headline that **states the answer** ("Three notices cover the water off Santa Cruz. The nearest sample is 12 days old.");
- a dateline;
- numbered margin notes that match numbered callouts on a framed map plate;
- a north arrow and a plate caption that names every encoding.

**Strengths:** the most distinctive, and the clearest link between text and place. It reads like a scientific briefing, which is what CoastWatch is.
**Weaknesses:**
- On phones the plate competes with the column for height.
- A fixed plate is less explorable than a full-bleed map.

## Scores

Scores are 1–5 and reflect my own judgement after building each direction; no users were tested. "Cost" is scored so that 5 means cheapest to build.

| | Usability | Clarity | Distinctiveness | Scientific accuracy | Cost | Total |
|---|---|---|---|---|---|---|
| A Observatory | 3 | 4 | 4 | 5 | 4 | 20 |
| B Living Coast | 4 | 4 | 3 | 3 | 3 | 17 |
| C Editorial Atlas | 3 | 5 | 5 | 4 | 3 | 20 |

## Recommendation: "Atlas reading, Observatory instrument"

The Explorer prototype (`#/explorer`) is this recommendation built out:

- **From the Atlas, for every reading surface:**
  - paper panels;
  - Newsreader headlines that state the answer;
  - the four-marks ledger, with each kind of knowing written as a sentence led by its mark;
  - numbered and threaded links between text and map (provenance threads are the interactive form of the Atlas callouts);
  - plate-style captions for legends.
- **From the Observatory, for the map:**
  - the dark sea with the depth tint;
  - exact cells;
  - class lines as the calm overview mode;
  - the honest "for / issued" time bar;
  - coordinates in mono, graticule in Research mode.
- **From Living Coast, only its restraint:**
  - generous spacing in Coastal mode;
  - one-time motion (ripple, thread draw-out), no loops;
  - plain-sentence Level 1.

  Its light basemap and hidden-outside lens are **not** adopted. The Explorer's lens keeps the surrounding cells at 20 % instead of hiding them.

**Why not pick one?**
- C alone fails the map-exploration tasks.
- A alone fails the general visitor and long official text.
- B alone weakens scientific clarity.

The combination maps cleanly onto the redesign's two modes. Coastal is Atlas with Living Coast restraint; Research is Atlas with more of the Observatory.
