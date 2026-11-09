# Ask Neritic: how it works, how ready it is

Track Q of the private pilot. The code is in `coastwatch-web/src/pilot/ask/**`, the page is `/ask`, and the endpoints are `/api/ask` and `/api/sms`. SMS registration is covered in [sms.md](sms.md).

## Readiness (decided from configuration only, `src/pilot/readiness.ts`)

| Mode | When | What answers |
|---|---|---|
| **Pilot demo** (now) | no `ANTHROPIC_API_KEY` | A **deterministic grounded answerer**. It quotes PortFact records and cites them. No AI model is involved. The page says so: "Pilot demo — answers are assembled from Neritic's data without an AI model". |
| **Live** | `ANTHROPIC_API_KEY` set | Claude writes the answer from the same PortFacts, and Neritic checks it before anyone sees it. On any failure it falls back to the deterministic answer, labelled "live mode fallback". |

Until the Today track's `src/pilot/facts.ts` is merged, the records come from a **typed fixture** (`src/pilot/ask/fixture.ts`). They are transcribed from plan §1 as of 2026-11-08. The page shows a **"Sample data"** badge and every answer says "Sample data — the pilot's fixture records, not today's published data".

## The pipeline (both modes)

1. **Safety pre-pass** (`intents.ts`, deterministic). Questions in these four groups never reach an answerer:
   - "Is it safe to eat / can I eat …";
   - "safe to go out / should I go out";
   - "cross the bar";
   - "is it open so I can fish / can I set pots / am I allowed".

   They get a fixed template (`templates.ts`) that points to CDPH and CDFW (seafood and closures) or to the NWS marine forecast and USCG (sea and bar). The related records follow, labelled "For information only". Neritic never says "safe".
2. **Place:**
   - A pilot port named in the question wins over the selected port.
   - A place outside the pilot (Crescent City, Bodega, Oregon, …) is declined.
   - With no port at all, Neritic asks which port.
3. **Retrieval:** exactly one port's `PortToday` (`source.ts`). Records whose text reads like an instruction (prompt injection in a data field) are withheld and counted in a caveat.
4. **Answer:**
   - **Demo:** an intent router covers status, toxin tests, sea/wind/tide, bloom, whale rules, what changed, sources and data age. Answers are composed only from fact wording. Every line cites its fact ids. Official records come first, and unverified official records always carry "not yet verified by a person".
   - **Live:**
     - Routing: deterministic first. Only questions it cannot place go to **Claude Haiku 5.5** (`claude-haiku-5-5`), which can only pick an intent.
     - Answering: **Claude Sonnet 5.5** (`claude-sonnet-5-5`, effort `low`) with structured output `{answer: [{text, citations}], citations, covered}`. The citation enum is the provided fact ids.
     - Rejection: an answer is rejected and regenerated once with the reasons if any citation doesn't resolve or any number or date isn't in its cited facts, or if any guard rule fails. A second failure falls back to the deterministic answer.
     - The system prompt (`live.ts`, `SYSTEM_PROMPT`) enforces the hierarchy, tells the model to say when the data doesn't cover the question, and tells it to treat fact text as data.
5. **Post-pass guard** (`guard.ts`, the TypeScript port of #49's `guard.py`). It runs on every answer, in both modes:
   - unknown citations;
   - uncited claims;
   - numbers and dates not in the cited facts;
   - safety claims;
   - chlorophyll presented as toxin;
   - observations presented as forecasts;
   - research overclaims;
   - the hierarchy;
   - "lifted" claims;
   - unverified records cited without saying so;
   - old measurements without a date.

   Failing lines are **removed, never rewritten**, and the answer says a line was withheld.
6. **System text** that no answerer writes:
   - data age (oldest and newest record, plus a stale note per record, using per-kind thresholds);
   - the first-exchange disclaimer;
   - official referrals;
   - the SMS rendering (≤ 2 segments, with a source tag).

## Evals

- `src/pilot/ask/evals/golden.json`: **44** golden questions. They cover all 4 ports, every intent, safety, out-of-area, need-place and off-topic.
- `src/pilot/ask/evals/adversarial.json`, three parts:
  - **26** end-to-end adversarial cases: safety bait, stale data, out-of-area, prompt injection in the question and inside data fields, and empty or unavailable data;
  - **19** scripted bad answers the guard must catch, ported from #49's 14 and extended;
  - **6** controls that must pass.
- They run in vitest against demo mode on every test run (`tests/unit/ask/evals.test.ts`): **all pass**.
- Live mode uses the same sets, **not run** (it costs money; it needs the owner's approval):

  ```
  ANTHROPIC_API_KEY=… NERITIC_ASK_DAILY_USD_CAP=2 npm run eval:ask:live
  ```

  Roughly 70 questions × about 4k input tokens on Sonnet 5.5 comes to well under $2 at current prices.

## Spend and abuse controls (live mode)

- **Daily cap:** `NERITIC_ASK_DAILY_USD_CAP` in USD per UTC day. The default is **$1** if unset. Each call is estimated before it is sent and metered from `usage` afterwards. At the cap, answers fall back to deterministic and say why.
- **Rate limits:**
  - web: 30 questions per 10 minutes per IP (`NERITIC_ASK_WEB_LIMIT`);
  - SMS: 10 per hour per number.
- **Instance limits:** limits and the meter are **per server instance**. On serverless hosting the real ceiling is cap × warm instances, so also set a **monthly spend limit on the Anthropic Console workspace**. That is the hard stop.

## Logging (off by default)

`AskLogger` is a no-op unless all of `NERITIC_ASK_LOG=supabase`, `SUPABASE_URL` (or `NEXT_PUBLIC_SUPABASE_URL`) and `SUPABASE_SERVICE_ROLE_KEY` are set. It records:

- the question, the retrieved and cited fact ids, the answer, mode, model id, intent, outcome and guard codes;
- a salted SHA-256 **hash** of the IP or phone number (`NERITIC_ASK_LOG_SALT`), never the raw value.

When logging is on, `/ask` says "Questions are logged for 90 days to check answers, then deleted". **Retention is 90 days.** The table and the scheduled delete belong in `supabase/**` (accounts track); the SQL to hand over:

```sql
create table public.ask_log (
  id bigint generated always as identity primary key,
  ts timestamptz not null default now(),
  channel text not null check (channel in ('web','sms')),
  mode text not null, model text, place_id text,
  question text not null, intent text not null, outcome text not null,
  retrieved_fact_ids text[] not null default '{}', cited_fact_ids text[] not null default '{}',
  answer text not null, guard_codes text[] not null default '{}', user_key text
);
alter table public.ask_log enable row level security; -- no policies: service role only
create index on public.ask_log (ts);
-- 90-day retention (pg_cron)
select cron.schedule('ask-log-retention', '17 4 * * *', $$delete from public.ask_log where ts < now() - interval '90 days'$$);
```

## Integration points

- **Today data (done):** `loadPortToday` in `src/pilot/ask/source.ts` reads `loadPortTodayFacts` (`src/pilot/today/portToday.ts`): the port's Today facts for every fishery with the sea, `dataset: "published"` unless the server is reading the test fixture ("Sample data"). The hand-written sample records are eval records only (`evals/sampleFacts.ts`); `evals/published.json` asks the golden questions of the real fact shape.
- **Today cards (done):** citation chips link to `/today?place=<id>#fact-<fact id>`; every fact on Today has that address, and arriving on one scrolls to its card and marks it.
- **Ask sheet:**

  ```tsx
  <AskSheet place="port-592" open={…} onClose={…} />
  ```

  It lives in `src/pilot/ask/ui/AskSheet.tsx` and opens Ask Neritic from the Today page's action.
- **Shell (done):** `/ask` renders in `PilotShell`, the same top bar, dock and Almanac styling as Today; ⋯ in the top bar links to it. Today opens the same chat in a sheet; "See it on Today" closes the sheet onto the card.
- **SMS store:** `SmsStore` (opt-out and home port) is in memory. It needs a durable implementation before a public number goes live (see [sms.md](sms.md)).

## Owner decisions and inputs

1. **Anthropic API key plus a spend-cap approval.** The plan suggests $30 for the week; set `NERITIC_ASK_DAILY_USD_CAP` (for example 4) and a Console workspace limit. Then approve one paid `npm run eval:ask:live` before showing live mode.
2. **Twilio:** account upgrade, then the sole-proprietor A2P 10DLC submission per [sms.md](sms.md). It needs the `/privacy` and `/sms-terms` URLs from Track A.
3. **Logging:** whether to turn it on for the pilot. If yes, the accounts track creates the `ask_log` table above.
