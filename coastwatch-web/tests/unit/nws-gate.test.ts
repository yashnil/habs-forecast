import { afterEach, describe, expect, it, vi } from "vitest";

// api.weather.gov's robots.txt disallows every automated client: without CW_NWS_API=1 the
// conditions route must not call it, and must point readers to NWS's own pages instead
describe("conditions route and NWS", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it("makes no request to api.weather.gov unless an operator enables it, and links to NWS instead", async () => {
    vi.stubEnv("CW_NWS_API", "");
    vi.stubEnv("CW_CONDITIONS_FIXTURE", "");
    const urls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (u: string) => {
        urls.push(String(u));
        return new Response("", { status: 404 });
      }),
    );
    const { GET } = await import("@/app/api/conditions/route");
    const res = await GET(new Request("http://x/api/conditions?lat=36.6&lon=-121.89"));
    const body = await res.json();
    expect(urls.some((u) => u.includes("api.weather.gov"))).toBe(false);
    expect(urls.some((u) => u.includes("ndbc.noaa.gov") || u.includes("tidesandcurrents.noaa.gov"))).toBe(true);
    expect(body.marine).toMatchObject({ status: "none", url: "https://www.weather.gov/marine/" });
    expect(body.alerts.status).toBe("none");
  });

  it("queries NWS when enabled", async () => {
    vi.stubEnv("CW_NWS_API", "1");
    vi.stubEnv("CW_CONDITIONS_FIXTURE", "");
    const urls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (u: string) => {
        urls.push(String(u));
        return new Response("", { status: 404 });
      }),
    );
    const { GET } = await import("@/app/api/conditions/route");
    await GET(new Request("http://x/api/conditions?lat=36.6&lon=-121.89"));
    expect(urls.some((u) => u.startsWith("https://api.weather.gov/"))).toBe(true);
  });
});
