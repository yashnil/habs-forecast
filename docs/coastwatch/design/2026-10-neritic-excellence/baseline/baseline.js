const {chromium}=require('playwright');const {BASE,bypassContext,watch}=require('./common2');
const OUT=process.env.OUT ?? 'baseline-shots';
const W=[[1440,900,false],[1024,768,false],[768,1024,true],[390,844,true],[320,700,true]];
const R=[['01-welcome','/',false,null],['02-map-forecast','/?region=monterey_bay&layer=forecast',false,null],['03-map-satellite','/?region=monterey_bay&layer=satellite',false,null],['04-map-currents','/?region=monterey_bay&layer=currents',false,null],
['05-mycoast','/coast?place=port-550&for=commercial',true,null],['06-hab-evidence','/coast?place=port-550&for=commercial',false,'hab'],['07-research','/coast?place=port-550&mode=research',true,null],
['08-bloom','/bloom',true,null],['09-fisheries','/fisheries',true,null],['10-about','/about',true,null],['11-sources','/sources',true,null],['12-crescent-dungeness','/coast?place=port-201&for=commercial&species=dungeness',true,null]];
(async()=>{const b=await chromium.launch();const rep=[];
for (const [w,h,mob] of W){const ctx=await bypassContext(b,{viewport:{width:w,height:h},isMobile:mob,hasTouch:mob,deviceScaleFactor:1});
 const p=await ctx.newPage();const errs=watch(p);
 for (const [n,path,full,act] of R){
  if(n==='02-map-forecast') await ctx.addInitScript(()=>{try{localStorage.setItem('cw-intro-dismissed-v1','1')}catch{}});
  try{await p.goto(BASE+path,{waitUntil:'load',timeout:60000});await p.waitForTimeout(5000);
  if(act==='hab'){const el=p.locator('[data-testid="hab-briefing"],[data-testid="hab-evidence"]').first();if(await el.count()){await el.scrollIntoViewIfNeeded();await p.waitForTimeout(800);await el.screenshot({path:`${OUT}/${w}-${n}.png`});continue;}}
  const sh=await p.evaluate(()=>document.documentElement.scrollHeight);
  await p.screenshot({path:`${OUT}/${w}-${n}.png`,fullPage:full&&sh<12000});rep.push(`${w} ${n} h=${sh}`);}catch(e){rep.push(`${w} ${n} ERR ${e.message.slice(0,80)}`);}
 }
 rep.push(`${w} errors=${errs.length}`);await ctx.close();}
await b.close();console.log(rep.join('\n'));})();
