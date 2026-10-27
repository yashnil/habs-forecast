"""Official-page watcher memory: a notification only when conditions appear, change or resolve.

Regression for issue #12 (October 2026): with the registry awaiting human review, every
6-hourly run commented the same alert on the review issue."""

from __future__ import annotations

import copy
import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

from coastwatch_pipeline import watch_official as wo
from coastwatch_pipeline.fixtures import FixtureFetcher, fixture_context
from coastwatch_pipeline.http import Response
from coastwatch_pipeline.sources import official

from .conftest import failing

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / ".github" / "scripts" / "official-review-issue.sh"
CDFW = "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories"
CDPH = "https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx"
T = ["2026-10-09T00:00:00Z", "2026-10-09T06:00:00Z", "2026-10-09T12:00:00Z", "2026-10-09T18:00:00Z", "2026-10-10T00:00:00Z", "2026-10-10T06:00:00Z"]


def _altered(url: str, transform):
    body = FixtureFetcher()(url).body.decode()
    return lambda u: Response(url=u, status=200, content_type="text/html", body=transform(body).encode())


def cdfw_edited(marker: str = "is now open in Humboldt County"):
    return FixtureFetcher(overrides={re.escape(CDFW): _altered(CDFW, lambda b: re.sub(r"remains closed in(\s|<[^>]+>)+Humboldt County", marker, b))})


def cdph_with_release(rid: str = "SN26-020", title: str = "CDPH Lifts Warning for Monterey County Bivalves"):
    inject = f'<a href="/Programs/OPA/Pages/{rid}.aspx">{title}</a>'
    return FixtureFetcher(overrides={re.escape(CDPH): _altered(CDPH, lambda b: b.replace("<body", inject + "<body", 1) if "<body" in b else inject + b)})


def cdfw_failing(msg: str = "HTTP 503"):
    return FixtureFetcher(overrides={re.escape(CDFW): failing(msg)})


class Runs:
    """Successive watcher runs sharing one state file, like successive workflow runs."""

    def __init__(self, tmp_path: Path, registry: Path = official.REGISTRY):
        self.state = tmp_path / "v1" / wo.STATE_FILE
        self.out = tmp_path / "o"
        self.registry = registry
        self.i = 0

    def __call__(self, fetcher=None, registry: Path | None = None) -> dict:
        now = T[self.i]
        self.i += 1
        return wo.run_watch(fixture_context(self.out, now=now, fetcher=fetcher), registry or self.registry, self.state, "https://example.test/run")


def registry_copy(tmp_path: Path, mutate) -> Path:
    raw = json.loads(official.REGISTRY.read_text())
    mutate(raw)
    p = tmp_path / "reg.json"
    p.write_text(json.dumps(raw))
    return p


# ---------------------------------------------------------------- repeated unchanged runs
def test_first_run_reports_once_then_repeated_unchanged_runs_do_not_notify(tmp_path):
    runs = Runs(tmp_path)
    before = official.REGISTRY.read_bytes()
    s1 = runs()
    # the committed registry awaits a human review: that is the one open condition
    assert list(s1["conditions"]) == ["review_pending"]
    assert s1["notify"] is True and s1["transitions"]["new"] == ["review_pending"]
    for s in (runs(), runs(), runs()):
        assert s["notify"] is False
        assert s["transitions"] == {"new": [], "updated": [], "resolved": [], "resolved_info": {}}
        assert s["marker"] == s1["marker"]
        assert s["conditions"]["review_pending"]["first_seen"] == s1["conditions"]["review_pending"]["first_seen"]
        assert s["conditions"]["review_pending"]["last_seen"] == s["updated_at"]
    assert runs.state.exists() and json.loads(runs.state.read_text())["schema_version"] == wo.STATE_SCHEMA_VERSION
    assert official.REGISTRY.read_bytes() == before  # the watcher never writes the registry


def test_unchanged_pages_that_already_differ_from_the_review_are_reported_once(tmp_path):
    runs = Runs(tmp_path)
    runs()
    s2 = runs(cdfw_edited())
    assert s2["transitions"]["new"] == ["changed:cdfw_health_advisories"]
    assert s2["notify"] is True
    c = s2["conditions"]["changed:cdfw_health_advisories"]
    assert c["category"] == "content_change" and c["detail"].startswith("sha256:")
    assert s2["sources"]["cdfw_health_advisories"]["content_changed_at"] == T[1]
    # the same difference on the next runs is the same condition: no new notification
    for s in (runs(cdfw_edited()), runs(cdfw_edited())):
        assert s["notify"] is False and s["marker"] == s2["marker"]
        assert s["sources"]["cdfw_health_advisories"]["content_changed_at"] == T[1]


# ---------------------------------------------------------------- new changes
def test_a_page_that_changes_again_is_an_update_not_a_repeat(tmp_path):
    runs = Runs(tmp_path)
    runs()
    s2 = runs(cdfw_edited("is now open in Humboldt County"))
    s3 = runs(cdfw_edited("is now open in Humboldt and Del Norte Counties"))
    assert s3["notify"] is True
    assert s3["transitions"]["updated"] == ["changed:cdfw_health_advisories"] and s3["transitions"]["new"] == []
    c = s3["conditions"]["changed:cdfw_health_advisories"]
    assert c["extra"]["previous_detail"] == s2["conditions"]["changed:cdfw_health_advisories"]["detail"]
    assert c["first_seen"] == T[1] and s3["sources"]["cdfw_health_advisories"]["content_changed_at"] == T[2]
    assert "Changed again" in wo.markdown(s3)


def test_new_cdph_release_and_its_lift_discrepancy_are_distinct_conditions_reported_once(tmp_path):
    runs = Runs(tmp_path)
    runs()
    s2 = runs(cdph_with_release())
    assert set(s2["transitions"]["new"]) == {
        "changed:cdph_release_list",
        "release:SN26-020",
        "discrepancy:SN26-020:cdph-2026-sn26-018-monterey-bivalves",
    }
    rel = s2["conditions"]["release:SN26-020"]
    assert rel["category"] == "awaiting_review" and "CDPH Lifts Warning" in rel["message"] and "SN26-020.aspx" in rel["message"]
    assert s2["sources"]["cdph_release_list"]["unreviewed_items"] == ["SN26-020"]
    s3 = runs(cdph_with_release())
    assert s3["notify"] is False and s3["marker"] == s2["marker"]
    md = wo.markdown(s3)
    assert "Nothing: the 4 open conditions below are unchanged" in md
    assert "SN26-020" in md and "Requires human review (4)" in md


# ---------------------------------------------------------------- network failures and recovery
def test_single_fetch_failure_is_transient_and_does_not_notify(tmp_path):
    runs = Runs(tmp_path)
    runs()
    s2 = runs(cdfw_failing("HTTP 503"))
    assert s2["notify"] is False and "unreadable:cdfw_health_advisories" not in s2["conditions"]
    src = s2["sources"]["cdfw_health_advisories"]
    assert src["ok"] is False and src["consecutive_failures"] == 1 and src["last_error"].startswith("HTTP 503")
    assert src["last_ok_at"] == T[0] and src["content_hash"]  # the last good reading is kept
    assert "transient" in wo.markdown(s2)


def test_repeated_fetch_failures_escalate_once_and_recovery_resolves(tmp_path):
    runs = Runs(tmp_path)
    runs()
    runs(cdfw_failing())
    s3 = runs(cdfw_failing("HTTP 403"))
    assert s3["transitions"]["new"] == ["unreadable:cdfw_health_advisories"] and s3["notify"] is True
    c = s3["conditions"]["unreadable:cdfw_health_advisories"]
    assert c["category"] == "fetch_failure" and "2 runs in a row" in c["message"] and "HTTP 403" in c["message"]
    # the error text changing is not a new condition
    s4 = runs(cdfw_failing("timed out"))
    assert s4["notify"] is False and s4["sources"]["cdfw_health_advisories"]["consecutive_failures"] == 3
    assert "Fetch failures (1)" in wo.markdown(s4)
    # readable again: resolved, reported once, then quiet
    s5 = runs()
    assert s5["transitions"]["resolved"] == ["unreadable:cdfw_health_advisories"] and s5["notify"] is True
    assert s5["transitions"]["resolved_info"]["unreadable:cdfw_health_advisories"]["resolution"] == "the page can be read again"
    assert s5["sources"]["cdfw_health_advisories"]["consecutive_failures"] == 0
    assert "**Resolved:** `unreadable:cdfw_health_advisories`" in wo.markdown(s5)
    assert runs()["notify"] is False


def test_fetch_failure_never_resolves_a_content_condition(tmp_path):
    runs = Runs(tmp_path)
    runs()
    s2 = runs(cdfw_edited())
    s3 = runs(cdfw_failing())
    assert s3["notify"] is False
    assert s3["conditions"]["changed:cdfw_health_advisories"] == {**s2["conditions"]["changed:cdfw_health_advisories"], "last_seen": s2["updated_at"]}
    assert s3["sources"]["cdfw_health_advisories"]["matches_review"] is False
    s4 = runs(cdfw_failing())
    assert s4["transitions"]["new"] == ["unreadable:cdfw_health_advisories"]
    assert "changed:cdfw_health_advisories" in s4["conditions"]


# ---------------------------------------------------------------- resolved notices
def test_release_recorded_as_reviewed_by_a_person_resolves_with_the_reason(tmp_path):
    runs = Runs(tmp_path)
    runs()
    runs(cdph_with_release())
    reviewed = registry_copy(tmp_path, lambda raw: raw["review"]["cdph_release_ids_reviewed"].insert(0, "SN26-020"))
    s3 = runs(cdph_with_release(), registry=reviewed)
    assert set(s3["transitions"]["resolved"]) == {"release:SN26-020", "discrepancy:SN26-020:cdph-2026-sn26-018-monterey-bivalves"}
    info = s3["transitions"]["resolved_info"]
    assert info["release:SN26-020"]["resolution"] == "recorded as reviewed in the registry"
    assert "changed:cdph_release_list" in s3["conditions"]  # the page still differs from the review fingerprint


def test_release_that_disappears_unreviewed_is_resolved_with_a_warning(tmp_path):
    runs = Runs(tmp_path)
    runs()
    runs(cdph_with_release())
    s3 = runs()
    assert "release:SN26-020" in s3["transitions"]["resolved"]
    assert "not reviewed" in s3["transitions"]["resolved_info"]["release:SN26-020"]["resolution"]


def test_human_review_clears_everything_and_the_all_clear_is_reported_once(tmp_path):
    runs = Runs(tmp_path)
    runs()
    verified = registry_copy(tmp_path, lambda raw: raw["review"].update(status="human_verified", reviewed_by="A Person", reviewed_at="2026-10-09T05:00:00Z"))
    s2 = runs(registry=verified)
    assert s2["conditions"] == {} and s2["notify"] is True
    assert s2["transitions"]["resolved"] == ["review_pending"]
    assert "a person recorded a review" in s2["transitions"]["resolved_info"]["review_pending"]["resolution"]
    md = wo.markdown(s2)
    assert "Requires human review (0)" in md and "Nothing is waiting for a person." in md
    s3 = runs(registry=verified)
    assert s3["notify"] is False and s3["conditions"] == {}


def test_history_records_every_transition_with_time_and_run(tmp_path):
    runs = Runs(tmp_path)
    runs()
    runs(cdfw_edited())
    s = runs()
    events = [(h["event"], h["key"], h["at"]) for h in s["history"]]
    assert events == [
        ("new", "review_pending", T[0]),
        ("new", "changed:cdfw_health_advisories", T[1]),
        ("resolved", "changed:cdfw_health_advisories", T[2]),
    ]
    assert all(h["run_id"] for h in s["history"])


# ---------------------------------------------------------------- report and CLI
def test_report_is_explicit_that_nothing_was_verified_and_links_every_source(tmp_path):
    runs = Runs(tmp_path)
    md = wo.markdown(runs(cdph_with_release()), "https://example.test/run")
    assert "edited nothing and verified nothing" in md and "cwp review-official" in md
    for src in official.load_registry().watched_sources:
        assert f"[{src.label}]({src.url})" in md
    assert "| Source | Last checked | Fetch | Content vs last review | Awaiting review |" in md
    assert "[run](https://example.test/run)" in md
    assert md.rstrip().endswith("-->") and wo.MARKER_PREFIX in md
    assert not re.search(r"\b(safe to eat|open to harvest|now open)\b", md, re.I)


def test_cli_writes_state_report_and_github_outputs(tmp_path, monkeypatch, capsys):
    from coastwatch_pipeline import cli

    monkeypatch.setattr(cli, "RunContext", lambda out_dir: fixture_context(tmp_path / "o", now=T[0]))
    state, md, gh = tmp_path / "official-watch.json", tmp_path / "report.md", tmp_path / "gh.txt"
    args = ["watch-official", "--state", str(state), "--markdown", str(md), "--github-output", str(gh), "--run-url", "https://example.test/run"]
    assert cli.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["notify"] is True and "report" in out and "history" not in out
    assert gh.read_text() == f"notify=true\nconditions=1\nmarker={out['marker']}\nchanged=true\n"
    assert "watcher report" in md.read_text() and f"<!-- {wo.MARKER_PREFIX} {out['marker']} -->" in md.read_text()
    persisted = json.loads(state.read_text())
    assert "report" not in persisted and persisted["marker"] == out["marker"]
    # second run with the same world: nothing to notify
    monkeypatch.setattr(cli, "RunContext", lambda out_dir: fixture_context(tmp_path / "o", now=T[1]))
    gh.unlink()
    assert cli.main(args) == 0
    assert gh.read_text().startswith("notify=false\nconditions=1\n")


def test_corrupt_or_old_state_is_ignored_not_fatal(tmp_path):
    runs = Runs(tmp_path)
    runs.state.parent.mkdir(parents=True)
    runs.state.write_text("{not json")
    assert runs()["notify"] is True
    runs.state.write_text(json.dumps({"schema_version": 99}))
    assert wo.load_state(runs.state) is None


# ---------------------------------------------------------------- the issue script (fake gh)
def _fake_gh(tmp_path: Path, open_issue: str = "", body: str = "", comments: list[str] | None = None) -> Path:
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (tmp_path / "comments.json").write_text(json.dumps(comments or []))
    gh = bin_ / "gh"
    gh.write_text(
        f"""#!/usr/bin/env bash
echo "$*" >> "{tmp_path}/gh.log"
case "$1 $2" in
  "issue list") echo "{open_issue}" ;;
  "api --paginate") python3 -c 'import json,sys; [print(c) for c in json.load(open(sys.argv[1]))]' "{tmp_path}/comments.json" ;;
  "api repos"*) python3 -c 'import json,sys; print(json.load(open(sys.argv[1])))' "{tmp_path}/body.json" ;;
  *) : ;;
esac
"""
    )
    (tmp_path / "body.json").write_text(json.dumps(body))
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
    return bin_


def _run_script(tmp_path: Path, bin_: Path, **env) -> list[str]:
    log = tmp_path / "gh.log"
    if log.exists():
        log.unlink()
    report = tmp_path / "report.md"
    report.write_text("report\n")
    r = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "GH_TOKEN": "x", "GITHUB_REPOSITORY": "o/r", "REPORT": str(report), **env},
        capture_output=True,
        text=True,
        check=True,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return [c for c in calls if c.startswith("issue ") and not c.startswith("issue list")] + [f"stdout: {r.stdout.strip()}"]


def test_issue_script_is_quiet_when_conditions_did_not_change(tmp_path):
    bin_ = _fake_gh(tmp_path, open_issue="12")
    calls = _run_script(tmp_path, bin_, NOTIFY="false", CONDITIONS="4", MARKER="abc")
    assert not any(c.startswith("issue comment") or c.startswith("issue create") for c in calls)
    assert "not commenting" in calls[-1]


def test_issue_script_opens_an_issue_when_none_is_open(tmp_path):
    bin_ = _fake_gh(tmp_path, open_issue="")
    calls = _run_script(tmp_path, bin_, NOTIFY="true", CONDITIONS="4", MARKER="abc")
    assert any(c.startswith("issue create") and "Official notices: sources changed since last review" in c for c in calls)
    assert not any(c.startswith("issue comment") for c in calls)


def test_issue_script_never_repeats_the_latest_report_even_without_state(tmp_path):
    older = f"<!-- {wo.MARKER_PREFIX} 1111 -->"
    latest = f"text\n<!-- {wo.MARKER_PREFIX} abc -->"
    bin_ = _fake_gh(tmp_path, open_issue="12", body=older, comments=[older, "a person's note", latest, "another note"])
    calls = _run_script(tmp_path, bin_, NOTIFY="true", CONDITIONS="4", MARKER="abc")
    assert not any(c.startswith("issue comment") for c in calls)
    assert "already the latest report on #12" in calls[-1]


def test_issue_script_comments_once_when_conditions_changed(tmp_path):
    bin_ = _fake_gh(tmp_path, open_issue="12", comments=[f"<!-- {wo.MARKER_PREFIX} 1111 -->"])
    calls = _run_script(tmp_path, bin_, NOTIFY="true", CONDITIONS="4", MARKER="abc")
    assert [c for c in calls if c.startswith("issue comment")] == [f"issue comment 12 --repo o/r --body-file {tmp_path}/report.md"]
    assert not any(c.startswith("issue close") for c in calls)


def test_issue_script_closes_the_issue_when_nothing_is_left_for_a_person(tmp_path):
    bin_ = _fake_gh(tmp_path, open_issue="12")
    calls = _run_script(tmp_path, bin_, NOTIFY="true", CONDITIONS="0", MARKER="def")
    assert any(c.startswith("issue comment 12") for c in calls) and any(c == "issue close 12 --repo o/r" for c in calls)
    # and does nothing when there is no issue to close
    bin_ = _fake_gh(tmp_path, open_issue="")
    calls = _run_script(tmp_path, bin_, NOTIFY="true", CONDITIONS="0", MARKER="def")
    assert not any(c.startswith("issue ") for c in calls)


def test_workflow_wires_the_watcher_state_report_and_script():
    wf = (REPO / ".github" / "workflows" / "coastwatch-data.yml").read_text()
    assert "--state ../publish/v1/official-watch.json" in wf and "--markdown ../watch-report.md" in wf
    assert "official_notify: ${{ steps.watch.outputs.notify }}" in wf
    assert "bash .github/scripts/official-review-issue.sh" in wf
    assert "name: watch-report" in wf
    assert "review-official" not in wf.split("notify-review:")[1].split("notify-health:")[0]  # the script, not the workflow, writes the body
