# Interview and usability-testing kit (pilot)

**Status:** this is a plan only. No one has been contacted, recruited or interviewed, and nothing below may be sent without the project owner's approval.

**How it relates to [#41's stakeholder research](../research/stakeholders-and-interviews.md):**
- §6 (30-minute interview scripts by group), §8 (the full 45–60-minute study) and §9 (consent and ethics) still apply. This kit does not repeat them.
- What it adds:
  - a **narrow first pilot**;
  - a **10–15-minute task** that fits on a dock or by phone;
  - recruitment drafts;
  - an observation sheet and friction log;
  - a novice-versus-experienced comparison;
  - a check on demand for "what changed" before anyone builds a notification service.

## 1. Pilot scope

| | |
|---|---|
| Primary group | **Commercial Dungeness crab fishermen on the North Coast.** Ports: Crescent City, Trinidad, Eureka; Fort Bragg as a contrast. Deckhands and skippers. |
| Secondary arm | **Recreational razor clam and mussel harvesters in Humboldt and Del Norte**: the same "is it open / what changed" question, but different agencies and a different toxin. |
| Workflow | Before the opener: "Is my area open, what is holding it, and what changed since I last looked?" Plus the harvester's "can I dig this weekend?" |
| Timing | Recruit in **late October and early November 2026**, before the Dec 1 opener and its pre-season testing. Avoid opener week and gear-setting days. Weather days are best. |
| n | **At least 5** in the primary group, **3–5** in the secondary arm. Add 3–5 more in a group if a critical error or a major, unresolved friction still appears in the last two sessions. |

## 2. Recruitment

Contact only through intermediaries the community trusts [S3]: California Sea Grant extension advisors (North Coast), port-based fishermen's associations, harbor district offices, tackle shops, and harvester-group moderators. Never contact anyone from licence lists. Recheck every organization name and contact on the day before using it.

**Message to an intermediary (draft, not sent):**

> Hi [name], I'm with CoastWatch, an independent research project (not a government agency) that gathers California's official shellfish and crab notices and ocean data in one place. Before the crab opener we'd like to watch 5–8 fishermen try a 15-minute task on their own phone, to see where it helps or gets in the way. We don't ask where anyone fishes, and we don't collect licence, permit or vessel numbers. Participants get [$ amount] for their time. Would you be willing to pass this note to people who might be interested? They contact us; we never contact them first.

**Note for participants (draft, not sent):**

> 15 minutes, by phone or at the harbor, at a time that suits you. You'll try to find a few things about crab or clam status using a test version of a website, while telling us what you're thinking. We're testing the website, not you. You'll get [$ amount] whether or not you finish. We don't record your name with your answers or ask where you fish. Reply to [project contact] if you're interested.

## 3. The 15-minute session

All tasks run on a **frozen test build**: a copy of production data fixed on a date, with a banner reading "Test copy, [date]: not current information". Nobody leaves with a stale or seeded status. Each participant does the **baseline task with their usual tools first**, then the CoastWatch tasks, so time-to-source can be compared within the same person.

| Min | Step | Script (read neutrally; never hint at the answer) |
|---|---|---|
| 0–2 | Consent (#41 §9) and context | "Which fisheries do you work, roughly which months? How do you usually find out whether crab is open?" Note role, years in the fishery, phone type and signal. Collect no locations. |
| 2–5 | **B: Baseline, usual tools** | "Using whatever you'd normally use, phone calls excepted, find out whether commercial Dungeness crab is open around [port], and show me where that comes from." Time stops when they open an agency source, or after 3 minutes. |
| 5–8 | **T-status, CoastWatch** | "Using this site, same question: is commercial Dungeness open around [port], and what is it based on?" *Success:* states the agency's season status and opens the agency source. *Critical errors:* says "open" when the season is closed or delayed; treats "no notice found" as open or safe; names CoastWatch as the authority. |
| 8–11 | **T-changed** | Set the device to "last visited 5 days ago" (seeded). "You last looked on [date]. What, if anything, has changed for your waters since then?" *Success:* names each seeded change and its date. *Critical error:* calls an older notice newly issued, or misses a lift or closure. **Run this task on both the current app and the prototype,** so "today" is not a straw man. |
| 11–13 | **T-gap** | Point to a hatched line, such as "No toxin sample within 40 km". "What does this tell you?" *Critical error:* reads it as "clean" or "safe". |
| 13–15 | Debrief | "What would you do next after this? What was missing? What, if anything, would stop you using this before a trip?" Then **correct any dangerous misunderstanding** and give the CDFW line (831) 649-2883 and the CDPH line (800) 553-4133. |

The **secondary arm** swaps in: "Can you dig razor clams at Clam Beach this weekend? Show the source." The seed is a summer reopening followed by a later closure. *Critical error:* uses the old reopening.

**Asking about demand without leading.** Never ask "Would you like alerts?" Ask instead:
- "Last time something changed for your fishery, how did you first hear about it, and how long after it happened?"
- "Walk me through the last time you checked whether something had changed."

Count spontaneous mentions of wanting to be told. Do not count agreement with a suggestion.

## 4. Observation sheet (one per participant per task)

| Field | Record |
|---|---|
| Participant code | P01… (no names) · group · novice or experienced (see §6) · device · signal (bars, or Wi-Fi) |
| Task | B / T-status / T-changed / T-gap / harvester |
| Start, first relevant answer, source opened (s) | Stopwatch timestamps |
| Taps · pages · external sites | Count |
| Outcome | Full / partial / fail, against the rubric in §3 |
| Critical error | Yes or no, with a verbatim quote |
| Confidence 1–7 | Asked after the task, before feedback |
| Stale or time-related understanding | Did they notice or mention the date of the information? (y/n + quote) |
| Observed vs. modelled understanding | If they mentioned forecast or chlorophyll: correct or confused (+ quote) |
| Friction log | One row per incident (§5) |

## 5. Friction and missing-information log

| # | Task | Where (page, element) | What happened (observed behaviour, not opinion) | Quote | Type | Severity | Missing information they asked for |
|---|---|---|---|---|---|---|---|

- **Type:** navigation, vocabulary, trust or source, time or freshness, geography or "is this my area", phone ergonomics (target size, glare, typing), connectivity, or missing information.
- **Severity:**
  - **critical:** a wrong safety, legal or status conclusion;
  - **major:** the task was blocked, or took more than 2× the baseline;
  - **minor:** a slowdown;
  - **cosmetic.**

After each session, add every incident to one shared, coded sheet. Repeated incidents are counted, not paraphrased.

## 6. Comparing novice and experienced users

**Who counts as which:**
- **Experienced:** at least 5 seasons in the fishery, and has made a decision affected by a closure.
- **Novice:** fewer than 2 seasons, or a visiting recreational harvester.
- Record this at screening. Aim for at least 2 novices in each group.

**How to compare:**
- Report **per person** (no percentages while n<8): task outcome, time to source, critical errors, and the gap between confidence and correctness.
- Expect experienced users to be faster at the baseline and possibly *more confident when wrong* (prior belief, H). Expect novices to be helped most by the plain-language status line (H).
- A design passes only if **neither** group makes a critical error on the status task.

## 7. Recording, privacy and consent

- **Consent:** as in #41 §9. Record on paper or verbally; audio is optional. Prefer screen recording on a test phone; use the participant's own phone only with consent, notifications hidden.
- **What is stored:** participant code, role category, broad region, timings, quotes. No names alongside data, and no licence, permit or vessel numbers, fishing locations or catch.
- **Storage:** codes are kept apart from any contact details, which are deleted after payment. Notes go in a private folder, never in the public repo. The public report contains only aggregated, coded findings.
- **Institutional review:** if a university or Sea Grant partner runs the study, its IRB decides whether review is needed before any contact.

## 8. What would change the plan

| Finding | Decision |
|---|---|
| Any critical error on T-status with the current or new design | Fix the wording before any wider release. Retest with 3 new participants. |
| T-changed beats the current app on outcome and time for ≥4 of 5 participants, **and** ≥3 spontaneously describe missing a change | Justifies the change list. Consider an opt-in notification only after that. |
| Participants rely on the hotline or colleagues and say they wouldn't check a website | Prioritise the printable or shareable dated briefing and the source links, not more features. |
| The baseline is already ≤60 s to the right source for most participants | CoastWatch's value is in "what changed" and the gaps, not in finding status. Rethink I1/I2. |

## 9. Consent sheet: draft for owner review (not used, nothing sent)

> **CoastWatch usability session: what you're agreeing to**
>
> **Who we are.** CoastWatch is an independent research project, not a government agency. We do not make, enforce or report on fishing rules.
>
> **What happens.** For about 15 minutes you try a few tasks on a test copy of our website and tell us what you're thinking. The test copy shows a banner with its date and is **not current information**. Please don't use it to decide anything.
>
> **What we record.** Your role (for example "commercial crab"), a broad region (for example "North Coast"), how long tasks take, and what you say. **Audio or screen recording only if you tick the box below.** We do not ask for your name alongside your answers. We do not collect licence, permit or vessel numbers, where you fish, or your catch.
>
> **Payment.** [amount, to be approved] for your time, **whether or not you finish**. It doesn't depend on what you say.
>
> **Your choices.** You can skip any task, stop at any time, or ask us to delete your notes within 30 days, by giving the participant code on this sheet.
>
> **Who sees it.** The research team only. Published results describe what groups of people did ("4 of 6 participants…"), never who said what. Quotes are used only without anything that could identify you.
>
> **After the session.** If anything we showed you could be misread, we'll tell you the correct official sources before you leave: CDFW (831) 649-2883, CDPH (800) 553-4133.
>
> ☐ I agree to take part ☐ Audio recording OK ☐ Screen recording OK ☐ Quotes without identifying details OK
> Participant code: P__ Date: ____

**Data-handling plan (for review):**

| Data | Where | Kept until | Who |
|---|---|---|---|
| Contact details (only to arrange the session and pay) | Kept apart from session notes; never in the repo | Deleted after payment, within 14 days | The person scheduling |
| Consent sheet (code only, no name) | Locked folder or encrypted drive | End of the project + 1 year, or the IRB's rule if one applies | Research team |
| Session notes, timings, quotes (code only) | Private, encrypted folder; never in the public repo | Until the analysis is published + 1 year | Research team |
| Recordings (only if ticked) | Same private folder | Deleted after transcription and coding, within 60 days | Research team |
| Published findings | The repo / PR | Permanent | Public; aggregated and coded only |

**Before anyone is contacted** (all need the owner's approval):
1. The incentive amount and how it is paid.
2. The intermediaries named in §2.
3. Whether a university or Sea Grant partner's IRB reviews the study.
4. Who holds the contact list.
