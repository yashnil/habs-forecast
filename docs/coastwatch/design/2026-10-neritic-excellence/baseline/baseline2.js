const {chromium}=require('playwright');const {BASE,bypassContext}=require('./common2');
const OUT=process.env.OUT ?? 'baseline-shots';
(async()=>{const b=await chromium.launch();
for (const [w,h,mob] of [[1440,900,false],[1024,768,false],[768,1024,true],[390,844,true],[320,700,true]]){const ctx=await bypassContext(b,{viewport:{width:w,height:h},isMobile:mob,hasTouch:mob});
 await ctx.addInitScript(()=>{try{localStorage.setItem('cw-intro-dismissed-v1','1')}catch{}});const p=await ctx.newPage();
 await p.goto(BASE+'/?region=monterey_bay&layer=multi',{waitUntil:'load'});await p.waitForTimeout(6000);await p.screenshot({path:`${OUT}/${w}-03-map-satellite.png`});
 if(w===1440){await p.goto(BASE+'/?region=monterey_bay&layer=forecast',{waitUntil:'load'});await p.waitForTimeout(5000);await p.mouse.click(560,420);await p.waitForTimeout(1500);await p.screenshot({path:`${OUT}/1440-13-map-click.png`});}
 if(w===390){await p.goto(BASE+'/coast?place=port-550&for=commercial',{waitUntil:'load'});await p.waitForTimeout(5000);const el=p.getByTestId('hab-evidence');if(await el.count()){await el.scrollIntoViewIfNeeded();await p.waitForTimeout(800);await p.screenshot({path:`${OUT}/390-06-hab-evidence-view.png`});}
  await p.goto(BASE+'/bloom',{waitUntil:'load'});await p.waitForTimeout(5000);await p.screenshot({path:`${OUT}/390-08-bloom-view.png`});}
 await ctx.close();}
await b.close();})();
