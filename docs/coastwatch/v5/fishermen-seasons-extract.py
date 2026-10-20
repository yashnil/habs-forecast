"""Neritic v5: extract CDFW regional sport-fishing summaries into coastwatch-web/src/content/seasons-cdfw.json.

Usage (read-only fetch, then two steps):
  for r in northern mendocino san-francisco central southern; do
    curl -s -A "Mozilla/5.0" -L https://wildlife.ca.gov/Fishing/Ocean/Regulations/Fishing-Map/$r > DIR/$r.html; done
  python3 -I fishermen-seasons-extract.py DIR            # writes DIR/regions.json
  python3 -I fishermen-seasons-extract.py DIR --emit > coastwatch-web/src/content/seasons-cdfw.json
Then update SEASONS_READ in src/content/seasons.ts. HTML comments are removed first: CDFW keeps old seasons inside them.
"""
import re, html, json, sys, os
d = sys.argv[1]
out = {}
def clean(s):
    s = re.sub(r'<br\s*/?>', ' ', s); s = re.sub(r'<[^>]+>', '', s); s = html.unescape(s)
    return re.sub(r'\s+', ' ', s).strip()
for r in ["northern", "mendocino", "san-francisco", "central", "southern"]:
    t = open(os.path.join(d, r + ".html"), encoding="utf-8", errors="replace").read()
    t = re.sub(r'<!--.*?-->', '', t, flags=re.S)
    upd = re.search(r'This summary of current regulations was updated on <strong>([^<]+)</strong>', t)
    i = t.find('Open Fishing Seasons'); j = t.find('Closed Fishing Seasons'); k = t.find('More Information', j)
    def items2(seg):
        res = []
        for m in re.finditer(r'<img alt="([^"]*)" src="[^"]*/(\w+)circle\.png"[^>]*/>(?:\s|&nbsp;)*<a href="#([^"]+)">(.*?)</a>', seg, re.S):
            res.append({"alt": m.group(1), "color": m.group(2), "anchor": m.group(3), "name": clean(m.group(4))})
        return res
    open_ = items2(t[i:j]); closed = items2(t[j:k])
    # section text: find anchor name and following paragraphs
    secs = {}
    for it in open_ + closed:
        a = it["anchor"]
        m = re.search(r'(?:name|id)="%s"[^>]*>' % re.escape(a), t[k:])
        if not m: continue
        seg = t[k + m.end(): k + m.end() + 8000]
        seg = seg[seg.find('</h3>') + 5:] if '</h3>' in seg[:400] else seg
        nxt = re.search(r'<h3|(?:name|id)="[a-z]+"', seg)
        seg = seg[: nxt.start()] if nxt else seg
        ps = [clean(x) for x in re.split(r'</?p[^>]*>', seg)]
        secs[a] = [p for p in ps if p][:3]
    out[r] = {"updated": upd.group(1) if upd else None, "open": open_, "closed": closed, "sections": secs}
json.dump(out, open(os.path.join(d, "regions.json"), "w"), indent=1)
if "--emit" not in sys.argv:
    sys.exit(0)
o = out
META = {
  "northern": ("Northern", "CA–OR border to 40°10′ N (near Cape Mendocino)", 40 + 10/60, 42.1),
  "mendocino": ("Mendocino", "40°10′ N (near Cape Mendocino) to 38°57.5′ N (Point Arena)", 38 + 57.5/60, 40 + 10/60),
  "san-francisco": ("San Francisco", "38°57.5′ N (Point Arena) to 37°11′ N (Pigeon Point)", 37 + 11/60, 38 + 57.5/60),
  "central": ("Central", "37°11′ N (Pigeon Point) to 34°27′ N (Point Conception)", 34 + 27/60, 37 + 11/60),
  "southern": ("Southern", "34°27′ N (Point Conception) to the U.S.–Mexico border", 32.4, 34 + 27/60),
}
regions = []
for rid, (label, bounds, south, north) in META.items():
  v = o[rid]
  items = []
  for lst, arr in (("open", v["open"]), ("closed", v["closed"])):
    for it in arr:
      ps = [p for p in v["sections"].get(it["anchor"], []) if p]
      mark = {"green": "open", "yellow": "alert", "red": "closed"}[it["color"]]
      items.append({"id": it["anchor"], "name": it["name"], "list": lst, "mark": mark, "quote": ps[0] if ps else ""})
  regions.append({"id": rid, "label": label, "bounds": bounds, "south": round(south, 4), "north": round(north, 4),
    "url": f"https://wildlife.ca.gov/Fishing/Ocean/Regulations/Fishing-Map/{rid}", "updated": v["updated"].replace(" ", " ").replace("&nbsp;", " "), "items": items})
print(json.dumps(regions, ensure_ascii=False, indent=1))
