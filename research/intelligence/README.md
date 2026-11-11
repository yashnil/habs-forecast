# CoastWatch Intelligence: evidence-grounded Q&A prototype (research only)

This is a read-only proof of concept. It answers fishermen's and harvesters' questions only from recorded CoastWatch data, and it checks every answer before showing it.

- It is not connected to the website, the pipeline or any LLM API.
- The model is a deterministic mock. A dry-run adapter builds the exact Claude request but refuses to send it.

```bash
cd research/intelligence
python3 -m cwi ask "Can I eat mussels I pick at Monterey this weekend?" --trace
python3 -m cwi eval                      # golden + adversarial evals; exit 0 only if all core checks pass
python3 -m cwi payload "Is there domoic acid near Santa Cruz?"   # Claude request body (never sent)
python3 -m unittest discover -s tests
```

It needs only the Python 3.10+ standard library: no packages, no network.

## How an answer is produced

```
question ─► provider (mock | scripted | Claude dry-run)
              │ tool_use blocks (Messages API shape)
              ▼
           Toolbox: 9 read-only tools over snapshots/2026-10-13
              │ evidence cards with stable ids + tier + flags
              ▼
           provider writes a draft that cites [evidence:id] after every factual sentence
              ▼
           guard.check(draft, retrieved evidence)  ── fail ─► one revision ─► fail ─► fallback (official notices only)
              ▼ pass
           answer + fixed footer ("Only CDPH and CDFW decide …")
```

**Evidence tiers.** These follow GOALS.md §2, and an answer must lead with tier 1 when a restrictive notice applies.

1. CDFW/CDPH notices and statements
2. CalHABMAP shore-station measurements
3. C-HARM model probabilities
4. Context: satellite chlorophyll, data status, landings, and the caveats published with the data
5. Offline research

**Tools.** Every tool reads only the recorded snapshot.

- `find_port`
- `get_official_notices`
- `get_measurements`
- `get_charm`
- `get_satellite_chlorophyll`
- `get_data_status`
- `get_fisheries_exposure`
- `get_research_results`
- `search_evidence` (BM25)

Port-to-notice relations reuse the pipeline's own `official_relations`; nothing is re-derived.

**Guard** ([cwi/guard.py](cwi/guard.py)). This is a deterministic check of the draft against the evidence actually returned in the conversation. It flags:

- **Citations:** an uncited factual sentence, or a citation to evidence that no tool returned.
- **Numbers and dates:** a number or date that isn't in the cited evidence. A percentage must be an evidence probability. An "issued/sampled" date must be that evidence's own date.
- **Safety and science claims:**
  - unnegated safety claims such as "safe to eat", "no risk" or "all clear";
  - chlorophyll presented as toxin;
  - a measurement-only sentence that makes a future claim;
  - the PINN described as beating the ConvLSTM.
- **The official hierarchy:** a restrictive notice that was retrieved but isn't cited, or an answer that doesn't lead with official information.
- **Notice status:**
  - a disputed notice cited without saying so;
  - anything described as "lifted";
  - official records cited without saying they aren't human-verified;
  - an old measurement cited without its date.

## Recorded data

`snapshots/2026-10-13/` is a byte-identical copy of the published `v1/` files from `coastwatch-data@c376845a`. Only the manifest is trimmed; see [SOURCE.md](snapshots/2026-10-13/SOURCE.md). The snapshot contains real conditions the assistant must handle:

- The Monterey County bivalve advisory (SN26-018) is **disputed**. CDPH release SN26-021 appears to lift it, but no person has reviewed that yet, so it must not be called active or lifted.
- The whole registry is `pending_human_review`.
- The latest C-HARM, satellite chlorophyll and HF-radar fetches **failed**.
- The Dungeness crab season is **closed**, even though there are no toxin closures.
- The latest Monterey Wharf domoic-acid sample is from 2022.

## Evaluation (`python3 -m cwi eval`)

| Set | What | Result (mock provider) |
|---|---|---|
| Golden, core (22) | Real questions: tools called, required evidence cited, required/forbidden text, abstention, guard | 22/22 pass, first-draft guard pass 100%, fallback 0% |
| Golden, hard (2) | Places that are not ports ("Channel Islands", "Tomales Bay") | 0/2. A known gap: place resolution covers ports and counties only |
| Adversarial (14) | Scripted bad answers, each with the violation code the guard must raise | 14/14 caught |
| Controls (5) | Acceptable answers that must not be flagged (negations, rounding, correct dispute wording) | 0 false positives |

**What these numbers do and do not show.**

- The mock and the guard were written together, so 22/22 shows the pipeline is wired correctly. It does not measure an LLM.
- The adversarial set is the meaningful test of the guard, but I wrote those cases too.
- The next step is to run the same golden set with a real model, which needs a paid call and your approval. Run each question several times, record the transcripts, then replay them through `ScriptedProvider`.

Two real bugs were found while building this, and both are now covered by tests:

- **Number leakage:** ISO dates were leaking their month and day into the numeric facts, so a fabricated "12%" passed.
- **Over-deletion:** revision could delete CDFW's "season is closed" sentence because only the last sentence of a quote was cited. Required-evidence recall now catches this.

## Limitations

- Place resolution covers port names, aliases and counties only (the hard cases above).
- The guard is lexical. A paraphrase can evade it, e.g. "nothing to worry about" in an unusual wording, or toxin implied without the word. An LLM-judge pass is the obvious addition; it should be evaluated against the same adversarial set and should never replace the deterministic checks.
- Numeric grounding checks a number against *some* value in the cited evidence, not against the right field. A model could cite the right card and swap two of its values.
- Single-turn only. There is no conversation memory or user location.

See [docs/integration-handoff.md](docs/integration-handoff.md) for how this would connect to the live product.
