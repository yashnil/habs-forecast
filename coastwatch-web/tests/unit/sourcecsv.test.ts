import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { sourceCsvUrl } from "@/lib/observations";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const obs = JSON.parse(readFileSync(path.join(FIX, readdirSync(FIX).find((f) => f.startsWith("observations-"))!), "utf8"));
const SC = "HABs-SantaCruzWharf";
const Q =
  'time,latitude,longitude,depth,Location_Code,SampleID,pDA,dDA,tDA,Pseudo_nitzschia_seriata_group,Pseudo_nitzschia_delicatissima_group,Avg_Chloro,Temp&time%3E=2014-01-01T00%3A00%3A00Z&orderBy("time")';
const url = (id: string, q = Q, host = "https://erddap.sccoos.org") => `${host}/erddap/tabledap/${id}.csv?${q}`;

describe("source CSV link (the pipeline's SCCOOS ERDDAP query)", () => {
  it("re-encodes every constraint character once, keeping the variable list readable", () => {
    expect(sourceCsvUrl({ station_id: SC, request_url: url(SC) })).toBe(
      "https://erddap.sccoos.org/erddap/tabledap/HABs-SantaCruzWharf.csv?time,latitude,longitude,depth,Location_Code,SampleID,pDA,dDA,tDA,Pseudo_nitzschia_seriata_group,Pseudo_nitzschia_delicatissima_group,Avg_Chloro,Temp&time%3E%3D2014-01-01T00%3A00%3A00Z&orderBy(%22time%22)",
    );
  });

  it("is idempotent: an already encoded query is not encoded twice", () => {
    const once = sourceCsvUrl({ station_id: SC, request_url: url(SC) })!;
    expect(sourceCsvUrl({ station_id: SC, request_url: once })).toBe(once);
    expect(once).not.toMatch(/%25/);
    expect(once).not.toMatch(/["<>\s]/);
  });

  it("refuses anything that is not a tabledap CSV request on SCCOOS for this station", () => {
    expect(sourceCsvUrl({ station_id: SC, request_url: undefined })).toBeNull();
    expect(sourceCsvUrl({ station_id: SC, request_url: "not a url" })).toBeNull();
    expect(sourceCsvUrl({ station_id: SC, request_url: url(SC, Q, "http://erddap.sccoos.org") })).toBeNull();
    expect(sourceCsvUrl({ station_id: SC, request_url: url(SC, Q, "https://erddap.sccoos.org.example.com") })).toBeNull();
    expect(sourceCsvUrl({ station_id: SC, request_url: url("HABs-MontereyWharf") })).toBeNull();
    expect(sourceCsvUrl({ station_id: SC, request_url: url(SC).replace(".csv?", ".json?") })).toBeNull();
    expect(sourceCsvUrl({ station_id: SC, request_url: url(SC, "") })).toBeNull();
    expect(sourceCsvUrl({ station_id: SC, request_url: url(SC, "pDA&time%3E=2014") })).toBeNull();
    expect(sourceCsvUrl({ station_id: "../x", request_url: url("../x") })).toBeNull();
    expect(sourceCsvUrl({ station_id: SC, request_url: url(SC, "time&%E0%A4%A") })).toBeNull();
  });

  it("gives every fixture station a link to its own dataset", () => {
    for (const s of obs.stations) {
      const u = sourceCsvUrl(s);
      expect(u, s.station_id).toMatch(new RegExp(`^https://erddap\\.sccoos\\.org/erddap/tabledap/${s.station_id}\\.csv\\?time,`));
    }
  });
});
