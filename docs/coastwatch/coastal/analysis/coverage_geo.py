# /// script
# dependencies = ["numpy", "shapely>=2.1,<3"]
# ///
"""Polygons of C-HARM coverage from the published nowcast grid (exact 0.03 deg cell squares):
toxin values, and the documented toxin mask (bloom value but no toxin value). Prototype only."""
import gzip, json
from pathlib import Path
import numpy as np, shapely
from shapely.geometry import box, mapping
H = Path(__file__).parent
g = lambda v: np.frombuffer(gzip.decompress((H / f"cur/lead0_{v}.u16.gz").read_bytes()), "<u2").reshape(391, 351) != 65535
pn, pda = g("pseudo_nitzschia"), g("particulate_domoic")
def cells(mask):
    I, J = np.nonzero(mask)
    return shapely.union_all([box(-127.5 + 0.03 * j - 0.015, 31.3 + 0.03 * i - 0.015, -127.5 + 0.03 * j + 0.015, 31.3 + 0.03 * i + 0.015) for i, j in zip(I, J)])
tox = cells(pda)
maskc = cells(pn & ~pda)
out = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"kind": "toxin_coverage_edge"}, "geometry": mapping(tox.boundary)},
    {"type": "Feature", "properties": {"kind": "toxin_mask"}, "geometry": mapping(maskc)},
]}
(H / "out" / "coverage_geo.json").write_text(json.dumps(out))
print((H / "out" / "coverage_geo.json").stat().st_size)
