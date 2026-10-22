import gzip,json,sys,os
import numpy as np
S=sys.argv[1]; D=S+'/v10-viz/data'
m=json.load(open(S+'/v10-viz-manifest.json'))
L=[l for l in m['layers'] if l['group_id']=='currents']
g=L[0]['vectors']['u_grid']; W,H=g['width'],g['height']
lat=g['lat_first']+np.arange(H)*g['lat_step']; lon=g['lon_first']+np.arange(W)*g['lon_step']
print('grid',W,H,'lat',lat[0],lat[-1],'lon',lon[0],lon[-1])
print('cell km: dy',g['lat_step']*111.32,'dx@36.8',g['lon_step']*111.32*np.cos(np.radians(36.8)))
def load(lid,c):
  a=np.frombuffer(gzip.decompress(open(f'{D}/{lid}.{c}.gz','rb').read()),dtype='<u2').reshape(H,W)
  f=a.astype(float)*g['scale_factor']+g['add_offset']; f[a==65535]=np.nan; return f
# Monterey bay box
mb=(lat>=36.55)&(lat<=37.0); mbx=(lon>=-122.25)&(lon<=-121.78)
hrs=[l for l in L if 'mean' not in l['layer_id']]
stack=[]
for l in hrs+[L[-1]]:
  u=load(l['layer_id'],'u'); v=load(l['layer_id'],'v'); sp=np.hypot(u,v); ok=np.isfinite(u)&np.isfinite(v)
  sub=ok[np.ix_(mb,mbx)]
  print(l['layer_id'][-12:], 'valid',ok.sum(), 'MB valid',sub.sum(),'/',sub.size, 'speed p50/p95/max %.2f/%.2f/%.2f'%(np.nanpercentile(sp,50),np.nanpercentile(sp,95),np.nanmax(sp)))
  if 'mean' not in l['layer_id']: stack.append(ok)
st=np.array(stack); n=st.sum(0)
print('cells ever valid',(n>0).sum(),'always valid',(n==len(hrs)).sum(),'valid>=12h',(n>=12).sum())
# flicker: cells changing state between consecutive hours
ch=[(st[i]!=st[i+1]).sum() for i in range(len(st)-1)]
print('state changes per hour step: mean',np.mean(ch),'max',max(ch))
# interior holes: invalid cells with >=3 of 4 valid neighbours in last hour
ok=st[-1]; p=np.pad(ok,1); nb=p[:-2,1:-1].astype(int)+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]
print('isolated holes (invalid, >=3 valid 4-nbrs):',((~ok)&(nb>=3)).sum(),' edge cells (valid, <4 valid nbrs):',(ok&(nb<4)).sum(),'of',ok.sum())
# where is coverage by latitude band
for a,b,name in [(32.4,34.6,'SoCal'),(34.6,36.0,'Central'),(36.0,37.2,'Monterey/Big Sur'),(37.2,38.3,'SF'),(38.3,40.0,'Mendo'),(40,42.1,'North')]:
  r=(lat>=a)&(lat<b); print(name, 'last hr valid', ok[r].sum(),' ever', (n>0)[r].sum())
# monterey bay finer: any coverage within 36.4-37.2 / -122.6..-121.7
r=(lat>=36.4)&(lat<=37.2); c=(lon>=-122.6)&(lon<=-121.7)
print('MB wide box per hour:', [int(s[np.ix_(r,c)].sum()) for s in st])
