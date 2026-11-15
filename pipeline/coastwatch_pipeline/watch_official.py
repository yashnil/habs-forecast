"""Official-page watcher with memory: conditions, transitions, and a per-source report.

The watcher itself (`sources/official.py`) is stateless: each run it reads the CDFW/CDPH
pages and compares them with the fingerprints a person recorded at the last review. Until a
person reviews again, every run finds the same differences. Reporting that as "changed"
each time is what filled the review issue with identical comments several times a day
(issue #12, October 2026).

This module keeps a small state file in the published dataset (`watch-official.json`,
carried from run to run like `health.json`) and turns each run's raw results into named
*conditions*. A notification is warranted only when the set of conditions changes:

- a condition appears        (a page changed since the review, a new CDPH release, a
                               release that appears to lift a record, a page that has been
                               unreadable for FAILURE_ESCALATION_RUNS runs in a row, ...)
- a condition changes        (a page that already differed from the review changed again)
- a condition resolves       (a person recorded a review, a release is no longer listed,
                               a page can be read again)

Three kinds of thing are kept apart, because they call for different responses:

- content changes            `changed:<source>`, `release:<id>`, `discrepancy:<id>:<record>`
                             -> a person must read the page and review the registry
- fetch failures             `unreadable:<source>`: a single failed run is recorded as
                             transient and not escalated; a page that cannot be read
                             FAILURE_ESCALATION_RUNS runs in a row (12 h on the 6-hourly
                             schedule) is. Recovery resolves it. A failed fetch never
                             resolves a content condition: the last reading is carried.
- awaiting human review      `review_pending` (the registry has never been verified by a
                             person) and the content conditions above until reviewed

Nothing here edits the registry or marks anything reviewed. Only a person running
`cwp review-official` after checking the sources can do that.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .context import RunContext
from .models import OfficialRegistry, WatchResult
from .sources import official

# Not "official-*.json": pipeline.prune() deletes files matching that pattern that the manifest does
# not reference (it owns the official-<hash>.json dataset files). The first name, official-watch.json,
# was pruned by every run, so no run ever saw the previous state (staging run 40466460080).
STATE_FILE = "watch-official.json"
STATE_SCHEMA_VERSION = 1
FAILURE_ESCALATION_RUNS = 2  # consecutive unreadable runs before a fetch failure is a condition
HISTORY_LIMIT = 60
MARKER_PREFIX = "neritic-official-watch:"
ISSUE_TITLE = "Official notices: sources changed since last review"
REVIEW_COMMAND = "cd pipeline && uv run cwp review-official --reviewer \"Your Name\" --confirm"

CATEGORIES = ("content_change", "fetch_failure", "awaiting_review", "registry")

_DISCREPANCY_RE = re.compile(r"CDPH release (SN\d\d-\d{3}) .*? active record (\S+)\. ")


def _short(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]


def _h8(content_hash: str | None) -> str:
    return (content_hash or "").removeprefix("sha256:")[:8] or "-"


@dataclass
class Condition:
    key: str
    kind: str
    category: str
    message: str
    source_id: str | None = None
    detail: str | None = None  # what counts as "the same condition"; a change here is an update
    first_seen: str = ""
    last_seen: str = ""
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = {
            "key": self.key,
            "kind": self.kind,
            "category": self.category,
            "message": self.message,
            "source_id": self.source_id,
            "detail": self.detail,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Condition":
        return cls(
            key=d["key"],
            kind=d["kind"],
            category=d["category"],
            message=d["message"],
            source_id=d.get("source_id"),
            detail=d.get("detail"),
            first_seen=d.get("first_seen", ""),
            last_seen=d.get("last_seen", ""),
            extra=d.get("extra") or {},
        )


@dataclass
class WatchInputs:
    """One run's raw observations."""

    results: list[WatchResult]
    registry_errors: list[str]
    conflicts: list[str]  # registry conflicts + reported lift discrepancies
    review_status: str
    reviewed_at: str
    reviewed_by: str
    reviewed_release_ids: list[str]
    item_titles: dict[str, str] = field(default_factory=dict)
    source_labels: dict[str, str] = field(default_factory=dict)
    release_source_id: str | None = None  # the source whose parser lists CDPH releases


def load_state(path: Path | None) -> dict | None:
    if not path or not path.exists():
        return None
    try:
        st = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return st if isinstance(st, dict) and st.get("schema_version") == STATE_SCHEMA_VERSION else None


# ---------------------------------------------------------------- conditions
def _conditions_now(inp: WatchInputs, prev_sources: dict, prev_conditions: dict[str, Condition], now: str) -> tuple[dict[str, Condition], dict, set[str]]:
    """The conditions observed in this run, the per-source status table, and the keys of
    conditions carried unchanged from the last run because their source could not be read."""
    conds: dict[str, Condition] = {}
    sources: dict[str, dict] = {}
    carried: set[str] = set()

    def add(c: Condition) -> None:
        conds[c.key] = c

    if inp.review_status != "human_verified":
        add(
            Condition(
                key="review_pending",
                kind="review_pending",
                category="awaiting_review",
                message=f"The registry has not been verified by a person: review status is `{inp.review_status}` (recorded {inp.reviewed_at} by {inp.reviewed_by}).",
                detail=inp.reviewed_at,
                first_seen=inp.reviewed_at,
            )
        )
    for e in inp.registry_errors:
        add(Condition(key=f"registry_error:{_short(e)}", kind="registry_error", category="registry", message=f"Registry invalid: {e}", detail=None))

    release_titles = inp.item_titles
    for c in inp.conflicts:
        if c.startswith(official.DISCREPANCY_PREFIX):
            m = _DISCREPANCY_RE.search(c)
            rid, record = (m.group(1), m.group(2)) if m else (_short(c), "?")
            add(
                Condition(
                    key=f"discrepancy:{rid}:{record}",
                    kind="discrepancy",
                    category="awaiting_review",
                    message=c,
                    source_id=inp.release_source_id,
                    detail=None,
                    extra={"release_id": rid, "record_id": record},
                )
            )
        else:
            add(Condition(key=f"conflict:{_short(c)}", kind="conflict", category="registry", message=c, detail=None))

    for w in inp.results:
        p = prev_sources.get(w.source_id) or {}
        label = inp.source_labels.get(w.source_id, w.source_id)
        s = {
            "label": label,
            "url": w.url,
            "last_checked": w.checked_at,
            "ok": w.ok,
            "http_status": w.http_status,
            "last_error": w.error if not w.ok else None,
            "consecutive_failures": 0 if w.ok else int(p.get("consecutive_failures") or 0) + 1,
            "last_ok_at": w.checked_at if w.ok else p.get("last_ok_at"),
            "content_hash": w.content_hash if w.ok else p.get("content_hash"),
            "content_changed_at": p.get("content_changed_at"),
            "matches_review": w.matches_review if w.ok else p.get("matches_review"),
            "unreviewed_items": list(w.new_items) if w.ok else list(p.get("unreviewed_items") or []),
        }
        if w.ok:
            if w.content_hash != p.get("content_hash"):
                s["content_changed_at"] = w.checked_at
            if w.matches_review is False:
                add(
                    Condition(
                        key=f"changed:{w.source_id}",
                        kind="page_changed",
                        category="content_change",
                        message=f"{label}: page content differs from the fingerprint recorded at the last review ({inp.reviewed_at}); current fingerprint {_h8(w.content_hash)}.",
                        source_id=w.source_id,
                        detail=w.content_hash,
                        extra={"url": w.url},
                    )
                )
            elif w.matches_review is None:
                add(
                    Condition(
                        key=f"unfingerprinted:{w.source_id}",
                        kind="unfingerprinted",
                        category="awaiting_review",
                        message=f"{label}: no fingerprint was recorded for this page at the last review; it cannot be compared until a person reviews.",
                        source_id=w.source_id,
                        detail=None,
                    )
                )
            for rid in w.new_items:
                title = release_titles.get(rid, "")
                add(
                    Condition(
                        key=f"release:{rid}",
                        kind="new_release",
                        category="awaiting_review",
                        message=f'{label}: release {rid} ("{title}", {official.CDPH_RELEASE_URL.format(rid=rid)}) is not in the reviewed list.' if title else f"{label}: release {rid} is not in the reviewed list.",
                        source_id=w.source_id,
                        detail=title or None,
                        extra={"release_id": rid, "url": official.CDPH_RELEASE_URL.format(rid=rid)},
                    )
                )
        else:
            # no new reading: carry this source's content conditions from the last run unchanged
            for k, c in prev_conditions.items():
                if c.source_id == w.source_id and c.kind != "unreadable" and k not in conds:
                    conds[k] = Condition.from_json(c.to_json())
                    carried.add(k)
            if s["consecutive_failures"] >= FAILURE_ESCALATION_RUNS:
                add(
                    Condition(
                        key=f"unreadable:{w.source_id}",
                        kind="unreadable",
                        category="fetch_failure",
                        message=f"{label}: could not be read for {s['consecutive_failures']} runs in a row (last error: {w.error}). Last successful read: {s['last_ok_at'] or 'never'}.",
                        source_id=w.source_id,
                        detail=None,
                        extra={"url": w.url, "consecutive_failures": s["consecutive_failures"]},
                    )
                )
        sources[w.source_id] = s
    return conds, sources, carried


def _resolution(c: Condition, inp: WatchInputs, sources: dict) -> str:
    if c.kind == "review_pending":
        return f"a person recorded a review ({inp.reviewed_at}, {inp.reviewed_by})"
    if c.kind == "page_changed":
        s = sources.get(c.source_id or "", {})
        return "the page matches the fingerprint of the recorded review" if s.get("matches_review") else "no longer reported"
    if c.kind == "new_release":
        rid = c.extra.get("release_id", "")
        return "recorded as reviewed in the registry" if rid in inp.reviewed_release_ids else "no longer listed on the page (not reviewed: check the registry)"
    if c.kind == "discrepancy":
        rid = c.extra.get("release_id", "")
        return "the release was reviewed or the record was edited by a person" if rid in inp.reviewed_release_ids else "no longer reported (the release is not listed, or the record changed)"
    if c.kind == "unreadable":
        return "the page can be read again"
    return "no longer reported"


def evaluate(inp: WatchInputs, previous: dict | None, now: str, run_id: str = "", run_url: str = "") -> dict:
    """Fold this run's observations into the previous state. Pure: no I/O."""
    prev = previous or {}
    prev_conditions = {k: Condition.from_json(v) for k, v in (prev.get("conditions") or {}).items()}
    prev_sources = prev.get("sources") or {}
    conds, sources, carried = _conditions_now(inp, prev_sources, prev_conditions, now)

    new, updated, resolved = [], [], []
    for k, c in conds.items():
        p = prev_conditions.get(k)
        if p is None:
            c.first_seen = c.first_seen or now
            c.last_seen = now
            new.append(k)
        else:
            c.first_seen = p.first_seen or c.first_seen or now
            c.last_seen = p.last_seen if k in carried else now  # a failed fetch is not an observation
            if c.detail != p.detail:
                c.extra["previous_detail"] = p.detail
                updated.append(k)
    resolved_info = {}
    for k, p in prev_conditions.items():
        if k not in conds:
            resolved.append(k)
            d = p.to_json()
            d["resolved_at"] = now
            d["resolution"] = _resolution(p, inp, sources)
            resolved_info[k] = d

    history = list(prev.get("history") or [])
    for k in new:
        history.append({"at": now, "run_id": run_id, "event": "new", "key": k, "message": conds[k].message})
    for k in updated:
        history.append({"at": now, "run_id": run_id, "event": "updated", "key": k, "message": conds[k].message})
    for k in resolved:
        history.append({"at": now, "run_id": run_id, "event": "resolved", "key": k, "message": resolved_info[k]["resolution"]})
    history = history[-HISTORY_LIMIT:]

    marker = hashlib.sha256("\n".join(f"{k}={conds[k].detail or ''}" for k in sorted(conds)).encode()).hexdigest()[:16]
    notify = bool(new or updated or resolved)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "updated_at": now,
        "run_id": run_id,
        "run_url": run_url,
        "review": {"status": inp.review_status, "reviewed_at": inp.reviewed_at, "reviewed_by": inp.reviewed_by},
        "sources": sources,
        "conditions": {k: conds[k].to_json() for k in sorted(conds)},
        "transitions": {"new": sorted(new), "updated": sorted(updated), "resolved": sorted(resolved), "resolved_info": resolved_info},
        "notify": notify,
        "marker": marker,
        "history": history,
    }


# ---------------------------------------------------------------- report
def _when(s: str | None) -> str:
    return (s or "-").replace("T", " ").removesuffix("Z") + ("Z" if s else "")


def markdown(state: dict, run_url: str = "") -> str:
    conds = state["conditions"]
    tr = state["transitions"]
    rv = state["review"]
    lines = ["## Official notices: watcher report", ""]
    head = f"Checked {_when(state['updated_at'])}"
    if run_url:
        head += f" · [run]({run_url})"
    head += f" · registry review: `{rv['status']}` (recorded {_when(rv['reviewed_at'])})"
    lines += [head, ""]
    lines += [
        "**This run edited nothing and verified nothing.** Official notices stay NOT verified until a person checks every record in "
        "`data/curated/official_notices.json` against its source and runs `" + REVIEW_COMMAND + "`.",
        "",
    ]

    lines += ["### What changed since the last run", ""]
    if not (tr["new"] or tr["updated"] or tr["resolved"]):
        n = len(conds)
        lines.append(f"Nothing: the {n} open condition{'s' if n != 1 else ''} below {'are' if n != 1 else 'is'} unchanged. No notification is due." if n else "Nothing: no open conditions. No notification is due.")
    for k in tr["new"]:
        lines.append(f"- **New:** {conds[k]['message']}")
    for k in tr["updated"]:
        c = conds[k]
        prev = c.get("extra", {}).get("previous_detail")
        suffix = f" (previously {_h8(prev)})" if c["kind"] == "page_changed" and prev else ""
        lines.append(f"- **Changed again:** {c['message']}{suffix}")
    for k in tr["resolved"]:
        r = tr["resolved_info"][k]
        lines.append(f"- **Resolved:** `{k}` ({r['resolution']}). Was: {r['message']}")
    lines.append("")

    lines += ["### Sources", "", "| Source | Last checked | Fetch | Content vs last review | Awaiting review |", "|---|---|---|---|---|"]
    for sid, s in state["sources"].items():
        if s["ok"]:
            fetch = f"ok ({s['http_status']})"
        else:
            n = s["consecutive_failures"]
            esc = "escalated" if n >= FAILURE_ESCALATION_RUNS else f"transient, escalates after {FAILURE_ESCALATION_RUNS}"
            fetch = f"**unreadable** ×{n} ({esc}); last read {_when(s['last_ok_at']) if s['last_ok_at'] else 'never'}"
        if s["matches_review"] is True:
            content = f"matches review ({_h8(s['content_hash'])})"
        elif s["matches_review"] is False:
            content = f"**changed since review** ({_h8(s['content_hash'])}, this content first seen {_when(s['content_changed_at'])})"
        else:
            content = "not compared"
        items = ", ".join(s["unreviewed_items"]) or "-"
        lines.append(f"| [{s['label']}]({s['url']}) | {_when(s['last_checked'])} | {fetch} | {content} | {items} |")
    lines.append("")

    review_needed = [c for c in conds.values() if c["category"] in ("content_change", "awaiting_review", "registry")]
    failures = [c for c in conds.values() if c["category"] == "fetch_failure"]
    lines += [f"### Requires human review ({len(review_needed)})", ""]
    if not review_needed:
        lines.append("Nothing is waiting for a person.")
    for c in review_needed:
        lines.append(f"- {c['message']} _(recorded since {_when(c["first_seen"])})_")
    lines.append("")
    if failures:
        lines += [f"### Fetch failures ({len(failures)})", ""]
        for c in failures:
            lines.append(f"- {c['message']} _(recorded since {_when(c["first_seen"])})_")
        lines.append("")
    lines.append(f"<!-- {MARKER_PREFIX} {state['marker']} -->")
    return "\n".join(lines)


# ---------------------------------------------------------------- run
def inputs_from_registry(ctx: RunContext, reg: OfficialRegistry) -> WatchInputs:
    errors, conflicts = official.validate_registry(reg)
    discrepancies: list[str] = []
    titles: dict[str, str] = {}
    results = official.watch(ctx, reg, discrepancies, item_titles=titles)
    release_src = next((s.id for s in reg.watched_sources if s.parser == "cdph_release_list"), None)
    return WatchInputs(
        results=results,
        registry_errors=errors,
        conflicts=conflicts + discrepancies,
        review_status=reg.review.status,
        reviewed_at=reg.review.reviewed_at,
        reviewed_by=reg.review.reviewed_by,
        reviewed_release_ids=list(reg.review.cdph_release_ids_reviewed),
        item_titles=titles,
        source_labels={s.id: s.label for s in reg.watched_sources},
        release_source_id=release_src,
    )


def run_watch(ctx: RunContext, registry: Path, state_path: Path | None, run_url: str = "") -> dict:
    """Read the pages, fold the results into the state file (if given), return the new state.
    The registry is never written."""
    reg = official.load_registry(registry)
    inp = inputs_from_registry(ctx, reg)
    state = evaluate(inp, load_state(state_path), ctx.now_iso, ctx.run_id, run_url)
    state["report"] = {
        "registry_errors": inp.registry_errors,
        "conflicts": inp.conflicts,
        "sources": [w.model_dump(exclude={"items_seen"}) for w in inp.results],
    }
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        persisted = {k: v for k, v in state.items() if k != "report"}
        state_path.write_text(json.dumps(persisted, indent=1, ensure_ascii=False) + "\n")
    return state
