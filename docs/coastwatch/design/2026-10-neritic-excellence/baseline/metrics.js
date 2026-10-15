const {chromium}=require('playwright');const {BASE,bypassContext}=require('./common2');
const R=[['map','/?region=monterey_bay&layer=forecast'],['mycoast','/coast?place=port-550&for=commercial'],['research','/coast?place=port-550&mode=research'],['bloom','/bloom'],['fisheries','/fisheries'],['about','/about']];
(async()=>{const b=await chromium.launch();const out={};
for (const [w,h,mob] of [[1440,900,false],[390,844,true]]){const ctx=await bypassContext(b,{viewport:{width:w,height:h},isMobile:mob,hasTouch:mob});
 await ctx.addInitScript(()=>{try{localStorage.setItem('cw-intro-dismissed-v1','1')}catch{}});const p=await ctx.newPage();
 for (const [n,u] of R){await p.goto(BASE+u,{waitUntil:'load'});await p.waitForTimeout(4500);
  out[w+' '+n]=await p.evaluate((mob)=>{const sizes={};let small=0,total=0;const fams=new Set();
   const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);let words=0;
   while(walker.nextNode()){const t=walker.currentNode;const s=t.textContent.trim();if(!s)continue;const el=t.parentElement;const cs=getComputedStyle(el);if(cs.visibility==='hidden'||cs.display==='none'||el.closest('[aria-hidden="true"]'))continue;const r=el.getBoundingClientRect();if(!r.width)continue;
    const fs=parseFloat(cs.fontSize);sizes[fs]=(sizes[fs]||0)+s.length;total+=s.length;if(fs<13)small+=s.length;fams.add(cs.fontFamily.split(',')[0]);words+=s.split(/\s+/).length;}
   const vis=[...document.querySelectorAll('a,button,[role=button],[role=tab],[role=radio],select,input')].filter(e=>{const r=e.getBoundingClientRect();return r.width>0&&r.height>0&&getComputedStyle(e).visibility!=='hidden'});
   const tiny=mob?vis.filter(e=>{const r=e.getBoundingClientRect();return r.height<44||r.width<44}).length:null;
   const vt=document.body.innerText;const nv=(vt.match(/not verified|unverified/gi)||[]).length;
   const firstScreen=[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return r.top<innerHeight&&r.bottom>0&&e.children.length===0&&e.textContent.trim()}).length;
   return {distinctSizes:Object.keys(sizes).length,sizes:Object.entries(sizes).sort((a,b)=>b[1]-a[1]).slice(0,8).map(([k,v])=>k+':'+Math.round(100*v/total)+'%').join(' '),pctUnder13:Math.round(100*small/total),fonts:[...fams].join('|'),words,controls:vis.length,controlsUnder44:tiny,notVerifiedMentions:nv,height:document.documentElement.scrollHeight};},mob);}
 await ctx.close();}
await b.close();console.log(JSON.stringify(out,null,1));})();
