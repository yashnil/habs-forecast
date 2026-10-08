"""Live end-to-end check against upstream servers. Run with: uv run pytest -m live"""

import pytest

from coastwatch_pipeline.context import RunContext
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.verify import verify_charm


@pytest.mark.live
def test_live_charm_matches_erddap(tmp_path):
    m = run_pipeline(RunContext(out_dir=tmp_path), ("charm",))
    st = next(s for s in m.sources if s.source_id == "charm")
    assert st.outcome in ("updated", "partial"), st.error
    report = verify_charm(tmp_path, live=True)
    assert report["summary"]["all_passed"], [r for r in report["rows"] if r["status"] == "FAIL"]
