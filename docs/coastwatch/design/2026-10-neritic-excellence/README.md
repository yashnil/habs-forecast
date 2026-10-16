# Neritic design review, October 2026

A proposal: nothing here changes the product. Start with [`review/index.html`](review/index.html) (open it in a browser). It covers:
- the audit;
- the reference study;
- the three logo directions;
- the two interface directions;
- the proposed tokens and motion rules;
- the recommendation and implementation phases.

| Folder | What |
|---|---|
| `review/` | The review page and its images: baseline screenshots, references, logo boards and prototype captures, as WebP. |
| `prototype-a/` | Direction A, scientific editorial. `build.py` writes the four pages from the published data of 15 Oct 2026 (`data/`). |
| `prototype-b/` | Direction B, precision instrument. The pages read `neritic-data.json`, extracted from the same published files. |
| `logo/` | Logo generator and the vectors in `out/` for A (Shelf), B (Depth N) and C (Soundings). |
| `research/references.md` | Notes on Emil Kowalski's skills, Impeccable, Taste Skill, Linear, Global Fishing Watch, Copernicus MyOcean and earth.nullschool. |
| `baseline/` | The scripts that captured and measured the current design, plus `metrics.json`. |

## Running the scripts
- **Prototypes** load MapLibre GL from unpkg and OpenFreeMap tiles, so they need a network connection. Serve the folder with any static server.
- **`logo/wm.mjs`** needs `opentype.js` and the OFL font files fetched from Google Fonts into `logo/fonts/`:

  | File | Font |
  |---|---|
  | `f2` | Sora 400 |
  | `f3` | Sora 500 |
  | `f4` | Sora 600 |
  | `f5` | Spectral 400 |
  | `f6` | Spectral 500 |
  | `f7` | Spectral 600 |
  | `f8` | Schibsted Grotesk 400 |
  | `f9` | Schibsted Grotesk 500 |
  | `f10` | Schibsted Grotesk 600 |

  Font binaries are not committed.
- **`baseline/`** scripts ran against the private preview through Vercel's protection-bypass cookie, using a small helper (`common2.js`: `BASE`, `bypassContext`, `watch`) that is not committed because it holds a deployment URL. Phase 0 of the plan replaces them with a Playwright visual-regression spec that runs in CI's Linux container.
