# Neritic Signal (formerly CoastWatch)

**Neritic Signal — Coastal intelligence, grounded in evidence.**

The product was called CoastWatch until October 2026. NOAA has run a program of that name for decades: **NOAA CoastWatch** (coastwatch.noaa.gov, and the West Coast node coastwatch.pfeg.noaa.gov), and this project uses its satellite data. The rename removes any suggestion that the project is NOAA's. "Neritic" is the oceanographic term for the shallow waters over the continental shelf, where the blooms, fisheries and harvests this project describes are. The new name also leaves room for the broader scope: fishing intelligence, environmental forecasting, research, and later an evidence-grounded assistant.

> Neritic Signal is an independent research project and is not affiliated with or endorsed by NOAA, CDFW, or CDPH.

## 1. Preliminary name screening (2026-10-15)
This is a preliminary screen, **not legal clearance**. Before buying a domain or filing anything, a person (ideally trademark counsel) should run a full search.

| Check | Finding |
|---|---|
| Exact phrase "Neritic Signal" (web) | No company, product, app, publication or project found. Results describe only the word *neritic*. |
| "Neritic" as a brand | No marine or ocean software company or app found. Unrelated ocean-software firms (CherSoft, TrueOcean, MariApps, OceanManager) use other names. |
| Science and environmental organisations | No NERITIC program or consortium. Nearby names: the French ANR project GALENE (*Global Assessment of Limnological, Estuarine and **Neritic** Ecosystems*, a descriptive word in its title) and the EU project NERITES (underwater cultural heritage). Neither is a conflict. |
| U.S. trademarks (third-party mirrors) | No NERITIC or NERITIC SIGNAL mark found. Near spellings: **NERETIC** (Reg. 6632577, Class 11, unrelated goods) and **NERITES** (2008 application, medical tissue sealants). USPTO's own search (tmsearch.uspto.gov) does not answer scripted queries, so it **was not checked directly**. Search NERITIC and NERITIC SIGNAL in Classes **9, 42 and 44** before any filing or purchase. |
| Domains (DNS and WHOIS, 2026-10-15) | neriticsignal.com, .org, .io: **not registered**. neriticsignal.app: no DNS (likely free; confirm at a registrar). neritic.com: registered 2002 and parked with a domain reseller (likely for sale). neritic.io: registered 2018, an unclaimed Squarespace placeholder. neritic.ai, neritic.app: not registered. |
| GitHub | `neriticsignal` and `neritic-signal`: free. `neritic`: an organisation from 2011 with one repository (a forum link). |
| App stores, social | No "Neritic" or "Neritic Signal" app or account found by search. Handles were not checked by signing in. |

**Assessment.** No serious conflict was found. "Neritic" on its own is a descriptive scientific word, so the protectable mark is the compound **Neritic Signal**, which appears unused.

Sources: [Merriam-Webster: neritic](https://www.merriam-webster.com/dictionary/neritic) · [Wikipedia: Neritic zone](https://en.wikipedia.org/wiki/Neritic_zone) · [ANR: GALENE](https://anr.fr/Project-ANR-23-MRS1-0002) · [GleanMark: NERETIC / NER* marks](https://gleanmark.com/search?q=NER) · [Justia trademarks](https://trademarks.justia.com/971/46/nereites-97146314.html)

## 2. What changed
- **Name and identity.** A new mark (`components/brand/BrandMark.tsx`) keeps the navy tile, cyan surface and amber signal, and draws the continental shelf in profile. The wordmark sets "Neritic" in semibold and "Signal" in regular.
- **Every user-facing surface:**
  - the masthead, footer and Welcome;
  - About and Data & sources;
  - My Coast, the Science tab and the HAB briefing;
  - the official notice texts;
  - Bloom, Fisheries and the map;
  - loading and outage states;
  - kept (printable) briefings, including the downloaded file name `neritic-signal-…html`;
  - API error text.
- **Browser and metadata:**
  - page titles (template `%s — Neritic Signal`);
  - description, Open Graph and Twitter cards, with a generated 1200×630 image (`app/opengraph-image.tsx`);
  - favicon, SVG icon, Apple touch icon;
  - web app manifest (`app/manifest.ts`) with 192, 512 and maskable icons.
  - Regenerate the icons with `node scripts/build_brand_assets.mjs`.
- **Independence statement:** About and Data & sources state it in full; the footer states it in one small line.
- **Published text written by the pipeline:**
  - the registry's source name;
  - the port-chlorophyll method;
  - tile notes;
  - the multi-sensor display's source and institution;
  - the fisheries tier note;
  - the health message;
  - the official-review issue body.
  - These values reach production on the first scheduled run after merge. **No schema or field changes.**
- **READMEs:** they introduce Neritic Signal and say "formerly CoastWatch" where that helps.

## 3. What did not change: NOAA CoastWatch
These keep their original words:
- every reference to **NOAA CoastWatch** as a data provider;
- its hosts (`coastwatch.noaa.gov`, `coastwatch.pfeg.noaa.gov`) and dataset ids;
- product names ("CoastWatch West Coast", "NOAA CoastWatch / NESDIS STAR", "CoastWatch ERDDAP").

The replacement was done line by line with exclusions for these, not as a global substitution, and `tests/e2e/brand.spec.ts` checks that /sources still credits NOAA CoastWatch.

Also unchanged:
- the paper's title, DOI and findings;
- historical documents under `docs/coastwatch/` and `research/`;
- release records and tags;
- logs and checkpoints.

## 4. Compatibility (staged rename)
| Area | Status |
|---|---|
| Saved settings | **Unchanged keys** (`cw-mode-v1`, `cw-intro-dismissed-v1`, remembered waters, kept briefings, and the rest), so nothing needs migrating and nothing is lost. A kept briefing saved before the rename now renders under the new name, with its original saved time and data date. |
| Links | Every route and query parameter is unchanged. Old shared links (`/?port=…`, `/coast?place=…&for=…&species=…`) open as before (tested). |
| APIs | `/api/conditions` and `/api/briefing`: same paths and JSON keys. Only the error text and the conditions User-Agent changed, to "Neritic Signal (formerly CoastWatch; …)". |
| Data and schemas | `schemas/v1` and generated types are unchanged, including their `https://coastwatch/schemas/v1/…` `$id`s. The data branches (`coastwatch-data`, `coastwatch-data-staging`), the data URL and every manifest field are unchanged. |
| Code | The `coastwatch-web/` directory and package, the `coastwatch_pipeline` module, the `cwp` CLI, `CW_*` environment variables and the `cw-` CSS and storage prefixes are all unchanged. |
| Infrastructure | The `coastwatch-demo` and `coastwatch-preview` Vercel projects, the public URL https://coastwatch-demo.vercel.app, the repository `yashnil/habs-forecast`, and the release and rollback tags are unchanged. |
| GitHub Actions | Workflow names ("CoastWatch CI", "CoastWatch data refresh"; another workflow triggers on the latter by name) and issue titles ("CoastWatch pipeline health", "CoastWatch data refresh failed"; found by exact title, so a rename would open duplicates) are unchanged. Issue **bodies** are rebranded. |
| NOAA CO-OPS request id | Tide requests keep `application=CoastWatch`, a client identifier NOAA's API asks for. It stays like the User-Agent, and the recorded test responses are keyed by the exact URL. |
| Pipeline User-Agent | `CoastWatch-pipeline/0.1 (+repo)` is unchanged while NOAA ERD investigates the runner HTTP 403s against it. Change it to `NeriticSignal-pipeline/0.1 (formerly CoastWatch-pipeline; +repo)` once they reply. |
| Monitoring | `health.json` keeps `schema: "coastwatch-health-1"`, and alert keys are unchanged, so alert history continues. |

## 5. Remaining occurrences (audit, 2026-10-15)
Case-insensitive "coastwatch" or "coast watch" in tracked and new files on `feat/neritic-signal-rebrand`, classified line by line:

| Count | Category | Why it stays |
|---:|---|---|
| 1679 | NOAA CoastWatch program, hosts, products, dataset ids | an external government program (rule above) |
| 405 | Historical documents and research records (`docs/coastwatch/`, `research/`, CLAUDE.md, GOALS.md) | history is not rewritten |
| 200 | Technical identifiers: directories, package and module names, schema `$id`, URLs, branches, Vercel projects | compatibility (§4) |
| 62 | Code comments and docstrings | internal, never displayed. They can be reworded opportunistically. |
| 37 | Workflow names, issue titles, data-branch names | jobs and issue search depend on them |
| 22 | Paths into `docs/coastwatch/`, the health schema id, a test origin, a docstring naming NOAA's ERDDAP | identifiers or NOAA |
| 7 | The internal change-group key `"coastwatch"` (SinceLastCheck) | an enum value, never displayed |
| 6 | Recorded upstream responses in test fixtures | NOAA's own bytes |
| 4 | Tests about the old name or NOAA | intended |
| 2 | The intended "formerly CoastWatch" statements | intended |

The rendered site has **no** old product name. `tests/e2e/brand.spec.ts` checks visible text and accessible labels on nine routes at 1440 and 390 px. The only allowances are NOAA's program names and the one "was called CoastWatch until October 2026" sentence on About.

**The research branch** (`research/coastwatch-intelligence`, PR #49) is untouched. If the evidence-grounded assistant ships, name it within the brand (for example "Neritic Signal Q&A"). Rename `research/intelligence` identifiers only when that work is integrated.

## 6. Domain strategy (recommendation, nothing bought)
1. Preferred canonical domain: **neriticsignal.com**, unregistered on 2026-10-15. Also register **neriticsignal.org** and **neriticsignal.app** defensively. Leave neritic.com (parked, likely costly) unless the shorter name becomes important.
2. Reserve the GitHub organisation `neriticsignal` and matching social handles before announcing the name.
3. **Move without breaking anything:**
   - Add the domain to the `coastwatch-demo` Vercel project as the primary domain. Keep `coastwatch-demo.vercel.app` serving the same deployment, or 308-redirecting with path and query preserved, so the BASES link and every shared link keep working.
   - Set `CW_SITE_URL` to the new origin for canonical and Open Graph URLs.
   - Rename the Vercel projects and the repository only after that, if at all. GitHub redirects renamed repositories, but the data URL `yashnil.github.io/habs-forecast/v1` is read by every deployed version, so keep it, or publish to both during a transition.
4. Run the USPTO search (§1) before buying.
