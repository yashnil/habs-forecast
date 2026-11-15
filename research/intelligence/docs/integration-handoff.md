# CoastWatch Intelligence: integration handoff (draft)

**Status:** research prototype on `research/coastwatch-intelligence`. Nothing here is wired into the site, the pipeline or a paid API. Every step below that touches production needs the owner's approval. This draft should be finalized once the release work (`release/2026-10-14`) and the fishermen-first UX work (`feat/fishermen-first-ux`, PR #47) have settled, because both change the surfaces an assistant would sit on.

## 1. What is ready to reuse

| Piece | Where | Reusable as-is? |
|---|---|---|
| Evidence model (ids, tiers, flags) | [cwi/evidence.py](../cwi/evidence.py) | The model is yes. The construction must move to TypeScript or the pipeline (see §2) |
| Tool contracts (names, JSON schemas, strict) | [cwi/tools.py](../cwi/tools.py) | Yes. These are the API surface a server route would expose |
| Guard rules | [cwi/guard.py](../cwi/guard.py) | Yes as a spec. Port it, or run it server-side in Python |
| System prompt | [cwi/agent.py](../cwi/agent.py) `SYSTEM_PROMPT` | Draft. Tune it against a real model with the eval |
| Golden and adversarial evals | [evals/](../evals) | Yes. Re-record expectations for each new snapshot |
| Claude request shape | `python3 -m cwi payload "…"` | Yes (claude-opus-5-5, adaptive thinking, `tool_choice: auto`, strict tools, cached system prompt) |

## 2. Decisions needed before integration (owner)

1. **Use one notice-applicability source.**
   - The site decides which notices relate to a place in `coastwatch-web/src/lib/applicability.ts`. It uses three tiers (geographic, unresolved, outside), and every association is "not reviewed" unless a person records a review.
   - The prototype instead uses the pipeline's `port-intel.official_relations`.
   - They can disagree, and an assistant that disagrees with the page it sits on is worse than none. **Recommendation:** have the assistant call the same applicability code the page uses, and carry its tier into the evidence flags, so the guard can require "may apply / does not settle it" wording for `unresolved`.
2. **Where the model call runs.** The site is static data plus client code; there is no API route today. A server route (a Vercel function) would hold the key and run the loop and the guard. That is new production infrastructure, with its own cost, rate limits and abuse handling, so it is an explicit approval item.
3. **Whether to ship an LLM at all for v1.** For ports, the deterministic writer ([MockProvider._write](../cwi/providers.py)) already produces grounded, guarded summaries without a model. A "port brief" built that way could ship first, with the free-text question box later. **Recommendation:** start with the deterministic brief.
4. **Logging and privacy.** Questions may reveal where someone fishes. Decide on retention, and whether transcripts are stored for evaluation.

## 3. Proposed path (each step reviewable on its own)

1. **Real-model evaluation, offline (needs approval for paid calls).**
   - Run the 24 golden questions through `claude-opus-5-5` with the prototype's tools, 3–5 samples each.
   - Record the transcripts and replay them with `ScriptedProvider`.
   - Report the first-draft guard pass rate, the revision and fallback rates, evidence recall, and the cost per answer.
   - Rough budget: about 25 × 5 × 2–3 calls.
2. **Expand the evals** with the questions in PR #47's fishermen-first research and the "unresolved" applicability cases. Add questions from at least one non-port place (beaches, islands).
3. **Port the guard and tools to the place the answer is rendered,** either TypeScript in the web app or a Python service. Use the same snapshot files the site loads (`manifest.json` and friends), so the assistant can never see different data from the page.
4. **Staging only.** Put it behind a flag on a preview deployment against the `staging` data channel. No production data and no `coastwatch-demo` changes.
5. **Human review gate.** Someone reads 50 real answers before any public exposure.

## 4. Invariants the integration must keep

- Official notices lead. A disputed notice is shown as disputed, and a notice is never "lifted" unless `cwp review-official` says so.
- No "safe" language. Chlorophyll is never toxin. Measurements are never forecasts.
- Research numbers are same-pipeline only. The PINN matches the ConvLSTM and does not beat it; see [research/pinn_audit](../../pinn_audit) for the open audit.
- The answer never adds facts that aren't on the page's data. The guard enforces this, and its fallback shows official notices only.
