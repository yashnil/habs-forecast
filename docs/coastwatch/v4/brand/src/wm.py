import os
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.boundsPen import BoundsPen
import sys, json
F={}
def font(n):
    if n not in F: F[n]=TTFont(os.path.join(os.path.dirname(os.path.abspath(__file__)),'fonts',n+'.ttf'))
    return F[n]
def outline(fn, text, size, track=0.0, adj=None, x0=0, base=0):
    """returns (d, width, bbox) ; size in units of em; y-down coordinates with baseline at base"""
    f=font(fn); cmap=f.getBestCmap(); gs=f.getGlyphSet(); upm=f['head'].unitsPerEm
    s=size/upm; x=x0; pen=SVGPathPen(gs, ntos=lambda v: ('%.2f'%v).rstrip('0').rstrip('.'))
    bp=BoundsPen(gs); adj=adj or {}
    for i,ch in enumerate(text):
        g=cmap[ord(ch)]
        tp=TransformPen(pen,(s,0,0,-s,x,base)); gs[g].draw(tp)
        tb=TransformPen(bp,(s,0,0,-s,x,base)); gs[g].draw(tb)
        x+=gs[g].width*s+track*size+adj.get(i,0)*size
    return pen.getCommands(), x-track*size, bp.bounds
if __name__=='__main__':
    a=json.loads(sys.argv[1]); d,w,b=outline(**a); print(json.dumps({'d':d,'w':w,'b':b}))
