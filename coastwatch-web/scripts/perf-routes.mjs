#!/usr/bin/env node
/**
 * Per-route performance, the method of docs/coastwatch/v12/research/E-science-and-perf.md section 4: installed
 * Chrome headless, cold load (cache disabled, service worker blocked), fixed settle; metrics from
 * PerformanceObserver (LCP, CLS, long tasks; TBT = long-task time over 50 ms after FCP); JS = CDP
 * encodedDataLength over the wire; map-ready = first frame window.__cwMap.loaded(). Conditions: desktop
 * unthrottled (3 runs), desktop and phone 390 at CPU 4x + DevTools Slow 4G (2 runs each).
 * `/` is a first visit (the opening plays); the other routes are a returning reader at Moss Landing following
 * Dungeness crab. Usage: npm run build, start the app on fixture data, then
 *   node scripts/perf-routes.mjs <baseUrl> <out.json>
 */
import { chromium } from "playwright";
import { writeFileSync } from "node:fs";
const BASE = process.argv[2] ?? "http://localhost:3410";
const OUT = process.argv[3] ?? "perf-results.json";
const REC = JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: "port-592", fisheries: ["dungeness_commercial"], vessel: null });
const ROUTES = [["home","/",false],["today","/today?place=port-592",true],["explore","/explore?place=port-592",true],["coast","/coast?place=port-592",true],["research","/research?place=port-592",true]];
const CONDS = {
  desktop: { vp:{width:1440,height:900}, cpu:1, net:null, settle:9000, runs:3 },
  desktop4x: { vp:{width:1440,height:900}, cpu:4, net:{offline:false,latency:562.5,downloadThroughput:1474560/8,uploadThroughput:675000/8}, settle:22000, runs:2 },
  phone4x: { vp:{width:390,height:844}, cpu:4, net:{offline:false,latency:562.5,downloadThroughput:1474560/8,uploadThroughput:675000/8}, settle:22000, runs:2 },
};
const results=[];
const browser = await chromium.launch({channel:"chrome",headless:true,args:["--use-angle=metal","--ignore-gpu-blocklist"]});
// PERF_CONDS=phone4x,desktop (a subset), PERF_ROUTES=today,explore, PERF_RUNS=1 for a quick look
const only = (k, v) => !process.env[k] || process.env[k].split(",").includes(v);
for (const [condName,c] of Object.entries(CONDS).filter(([k])=>only("PERF_CONDS",k))) for (const [name,path,seed] of ROUTES.filter(([n])=>only("PERF_ROUTES",n))) for (let run=0;run<(Number(process.env.PERF_RUNS)||c.runs);run++){
  const ctx = await browser.newContext({viewport:c.vp,serviceWorkers:"block",deviceScaleFactor:1});
  const page = await ctx.newPage();
  if (seed) await page.addInitScript((r)=>{localStorage.setItem("nx-port-v1",r);localStorage.setItem("nx-tour-v1","dismissed");localStorage.setItem("nx-opening-seen-v1","1")},REC);
  await page.addInitScript(()=>{
    window.__perf={lcp:0,lcpEl:null,lcpText:0,lcpTextEl:null,cls:0,shifts:[],long:[],fcp:0};
    new PerformanceObserver(l=>{for(const e of l.getEntries()){window.__perf.lcp=e.startTime;if(e.element&&!/^(IMG|CANVAS|IMAGE|VIDEO|svg)$/i.test(e.element.tagName)){window.__perf.lcpText=e.startTime;window.__perf.lcpTextEl=e.element.tagName+"|"+(e.element.textContent||"").slice(0,40)}window.__perf.lcpEl=(e.element?e.element.tagName+"."+(e.element.className||"").toString().slice(0,50)+"|"+(e.element.textContent||"").slice(0,40):e.url)}}).observe({type:"largest-contentful-paint",buffered:true});
    new PerformanceObserver(l=>{for(const e of l.getEntries())window.__perf.long.push([Math.round(e.startTime),Math.round(e.duration)])}).observe({type:"longtask",buffered:true});
    new PerformanceObserver(l=>{for(const e of l.getEntries())if(!e.hadRecentInput){window.__perf.cls+=e.value;if(e.value>0.01)window.__perf.shifts.push([Math.round(e.startTime),+e.value.toFixed(3),(e.sources||[]).map(s=>s.node?(s.node.className||s.node.nodeName||"").toString().slice(0,40):"?").join(",")])}}).observe({type:"layout-shift",buffered:true});
    const t0=performance.now();
    const tick=()=>{ if(window.__cwMap&&window.__cwMap.loaded&&window.__cwMap.loaded()&&!window.__perf.map){window.__perf.map=performance.now()-t0;return} if(performance.now()-t0<60000) requestAnimationFrame(tick)};
    requestAnimationFrame(tick);
  });
  const cdp = await ctx.newCDPSession(page);
  await cdp.send("Network.enable");
  await cdp.send("Network.setCacheDisabled",{cacheDisabled:true});
  if(c.cpu>1) await cdp.send("Emulation.setCPUThrottlingRate",{rate:c.cpu});
  if(c.net) await cdp.send("Network.emulateNetworkConditions",c.net);
  const reqs=new Map(); const bytes={js:0,css:0,font:0,img:0,doc:0,data:0,other:0}; let jsFiles=0, n=0;
  cdp.on("Network.responseReceived",e=>reqs.set(e.requestId,{url:e.response.url,mime:e.response.mimeType,type:e.type}));
  cdp.on("Network.loadingFinished",e=>{const r=reqs.get(e.requestId);if(!r)return;n++;const b=e.encodedDataLength;
    if(r.type==="Script"||/javascript/.test(r.mime)){bytes.js+=b;jsFiles++}
    else if(r.type==="Stylesheet")bytes.css+=b; else if(r.type==="Font")bytes.font+=b; else if(r.type==="Image")bytes.img+=b; else if(r.type==="Document")bytes.doc+=b; else if(r.type==="Fetch"||r.type==="XHR")bytes.data+=b; else bytes.other+=b;});
  let err=null;
  try{ await page.goto(BASE+path,{waitUntil:"load",timeout:90000}); }catch(e){err=String(e).slice(0,100)}
  await page.waitForTimeout(c.settle);
  const p = await page.evaluate(()=>{const nav=performance.getEntriesByType("navigation")[0];const fcp=performance.getEntriesByName("first-contentful-paint")[0];
    return {...window.__perf,fcp:fcp?fcp.startTime:null,ttfb:nav?nav.responseStart:null,load:nav?nav.loadEventEnd:null,dcl:nav?nav.domContentLoadedEventEnd:null, hasMap:!!window.__cwMap}});
  const after=p.fcp||0; const tbt=p.long.reduce((a,[s,d])=>a+(s+d>after?Math.max(0,d-50):0),0);
  const row={cond:condName,route:name,path,run,ttfb:Math.round(p.ttfb),fcp:Math.round(p.fcp),lcp:Math.round(p.lcp),lcpEl:p.lcpEl,lcpText:Math.round(p.lcpText),lcpTextEl:p.lcpTextEl,shifts:p.shifts,cls:+p.cls.toFixed(4),tbt:Math.round(tbt),longN:p.long.length,longMax:p.long.reduce((a,[,d])=>Math.max(a,d),0),load:Math.round(p.load),mapReady:p.map?Math.round(p.map):null,hasMap:p.hasMap,kb:Object.fromEntries(Object.entries(bytes).map(([k,v])=>[k,Math.round(v/1024)])),jsFiles,requests:n,err};
  results.push(row); console.log(JSON.stringify(row)); writeFileSync(OUT,JSON.stringify(results,null,1));
  await ctx.close();
}
await browser.close();
