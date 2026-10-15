"""Time budgets and the per-host hang breaker: one hanging upstream cannot hold a refresh past
the job's time limit (run 38116032798, 2026-10-11: SCCOOS's ERDDAP hung on every CalHABMAP
station, ~400 s each, and the job was cancelled at 40 minutes with nothing published)."""

from __future__ import annotations

import socket
import urllib.error
import urllib.request

import pytest

from coastwatch_pipeline import http


class Clock:
    """Fake monotonic clock: requests and sleeps advance it instead of waiting."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(http.time, "monotonic", c.monotonic)
    monkeypatch.setattr(http.time, "sleep", c.sleep)
    yield c
    http.clear_budgets()


def hanging(clock, calls):
    """urlopen that waits its whole timeout, then times out (a server that never answers)."""

    def urlopen(req, timeout, context):
        calls.append((req.full_url, timeout))
        clock.t += timeout
        raise urllib.error.URLError(socket.timeout("timed out"))

    return urlopen


def test_without_a_budget_fetch_behaves_as_before(clock, monkeypatch):
    calls: list = []
    monkeypatch.setattr(urllib.request, "urlopen", hanging(clock, calls))
    with pytest.raises(http.FetchError, match="Failed after 4 attempts") as e:
        http.fetch("https://slow.test/erddap/tabledap/A.csv?x")
    assert len(calls) == 4 and all(t == 90.0 for _, t in calls)
    assert e.value.hang and not e.value.budget


def test_a_used_up_budget_fails_fast_without_a_request(clock, monkeypatch):
    calls: list = []
    monkeypatch.setattr(urllib.request, "urlopen", hanging(clock, calls))
    http.start_budget("calhabmap", 60)
    clock.t += 61
    with pytest.raises(http.FetchError, match="calhabmap time budget \\(60 s\\) used up; not requested") as e:
        http.fetch("https://slow.test/a")
    assert calls == [] and e.value.budget


def test_timeouts_and_retries_never_run_past_the_budget(clock, monkeypatch):
    calls: list = []
    monkeypatch.setattr(urllib.request, "urlopen", hanging(clock, calls))
    http.start_budget("calhabmap", 120)
    t0 = clock.t
    with pytest.raises(http.FetchError, match="used up") as e:
        http.fetch("https://slow.test/a")
    assert clock.t - t0 <= 120
    assert calls[0][1] == 90.0 and all(t <= 120 for _, t in calls)
    assert e.value.hang and e.value.budget


def test_the_nearer_of_the_source_and_run_budgets_applies(clock, monkeypatch):
    calls: list = []
    monkeypatch.setattr(urllib.request, "urlopen", hanging(clock, calls))
    http.start_budget("run", 50, scope="run")
    http.start_budget("satellite_chl", 900)
    with pytest.raises(http.FetchError, match="run time budget"):
        http.fetch("https://slow.test/a")
    assert calls[0][1] == 50


def test_a_server_that_answers_is_not_a_hang(clock, monkeypatch):
    def unavailable(req, timeout, context):
        raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    with pytest.raises(http.FetchError) as e:
        http.fetch("https://busy.test/a")
    assert not e.value.hang


def test_host_breaker_skips_a_silent_host_but_not_its_neighbours(clock, monkeypatch):
    calls: list = []
    monkeypatch.setattr(urllib.request, "urlopen", hanging(clock, calls))
    cb = http.CircuitBreaker()
    for station in ("HABs-TrinidadPier", "HABs-Humboldt"):
        with pytest.raises(http.FetchError):
            cb(f"https://erddap.sccoos.test/erddap/tabledap/{station}.csv?time")
    n = len(calls)
    with pytest.raises(http.FetchError, match="skipped for the rest of this run") as e:
        cb("https://erddap.sccoos.test/erddap/tabledap/HABs-SantaCruzWharf.csv?time")
    assert len(calls) == n and e.value.hang  # no request was made

    # another host is unaffected
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout, context: _ok())
    assert cb("https://coastwatch.noaa.test/erddap/griddap/x.csv").body == b"ok"


def test_host_breaker_ignores_refusals_and_resets_on_success(clock, monkeypatch):
    state = {"mode": "hang"}
    calls: list = []
    hang = hanging(clock, calls)

    def urlopen(req, timeout, context):
        if state["mode"] == "hang":
            return hang(req, timeout, context)
        if state["mode"] == "403":
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)
        return _ok()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    cb = http.CircuitBreaker()
    with pytest.raises(http.FetchError):
        cb("https://h.test/erddap/griddap/A.csv")  # one hang
    state["mode"] = "ok"
    assert cb("https://h.test/erddap/griddap/B.csv").body == b"ok"  # answers: count resets
    state["mode"] = "hang"
    with pytest.raises(http.FetchError):
        cb("https://h.test/erddap/griddap/C.csv")
    state["mode"] = "403"  # NOAA refuses one dataset: an answer, never a host outage
    for d in ("D", "E", "F"):
        with pytest.raises(http.FetchError, match="Failed after 4 attempts"):
            cb(f"https://h.test/erddap/griddap/{d}.csv")
    assert "h.test" not in cb.hosts_down


def test_sccoos_outage_replayed_stays_inside_the_source_budget(clock, monkeypatch):
    """Twelve stations on one silent host: the old code spent ~400 s per station (80 min);
    now the host is skipped after two and the source never exceeds its budget."""
    calls: list = []
    monkeypatch.setattr(urllib.request, "urlopen", hanging(clock, calls))
    cb = http.CircuitBreaker()
    http.start_budget("calhabmap", 300)
    t0 = clock.t
    failed = 0
    for i in range(12):
        with pytest.raises(http.FetchError):
            cb(f"https://erddap.sccoos.test/erddap/tabledap/HABs-Station{i}.csv?time")
        failed += 1
    assert failed == 12
    assert clock.t - t0 <= 300
    assert len(calls) <= 8


def _ok():
    class R:
        status = 200
        headers = {"Content-Type": "text/csv"}
        left = [b"ok"]

        def read(self, n=-1):
            return self.left.pop() if self.left else b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return R()


def test_a_host_that_refuses_several_datasets_is_not_asked_again_this_run(clock, monkeypatch):
    # runs 38328551760, 38480042565, 38513961299: coastwatch.pfeg.noaa.gov answered every request
    # with 403 from the first one on, while other hosts answered; ~40-95 s of retries per dataset
    calls: list = []

    def urlopen(req, timeout, context):
        calls.append(req.full_url)
        if "refusing.test" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {"Server": "Apache"}, None)
        return _ok()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    cb = http.CircuitBreaker()
    for d in ("A", "B", "C"):
        with pytest.raises(http.FetchError, match="Server: Apache") as ei:
            cb(f"https://refusing.test/erddap/griddap/{d}.csv")
        assert ei.value.refused
    n = len(calls)
    with pytest.raises(http.FetchError, match="not asked again this run"):
        cb("https://refusing.test/erddap/griddap/D.csv")
    assert len(calls) == n  # no request at all
    assert cb("https://other.test/erddap/griddap/A.csv").body == b"ok"  # other hosts unaffected
    assert "refusing.test" not in cb.hosts_down  # a refusal is not a hang


def test_a_refusal_mixed_with_other_errors_is_not_counted_as_a_refusal(clock, monkeypatch):
    codes = iter([403, 502, 403, 403] * 3)

    def urlopen(req, timeout, context):
        raise urllib.error.HTTPError(req.full_url, next(codes), "x", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    cb = http.CircuitBreaker()
    for d in ("A", "B", "C"):
        with pytest.raises(http.FetchError) as ei:
            cb(f"https://h.test/erddap/griddap/{d}.csv")
        assert not ei.value.refused
    assert "h.test" not in cb.hosts_refusing
