import { NextResponse } from "next/server";
import { allow, askDataset, meta, serveAsk } from "@/pilot/ask/service";
import { suggestionsFor } from "@/pilot/ask/suggestions";
import type { AskRequest, SeenSnapshot } from "@/pilot/ask/types";

export const dynamic = "force-dynamic";

const NO_STORE = { "Cache-Control": "no-store" };

/** GET /api/ask?place=port-592 → readiness, logging and suggested questions (no answer). */
export async function GET(req: Request) {
  const place = new URL(req.url).searchParams.get("place");
  return NextResponse.json({ ...meta(process.env, await askDataset()), suggestions: suggestionsFor(place) }, { headers: NO_STORE });
}

function clientKey(req: Request): string {
  return (req.headers.get("x-forwarded-for") ?? "").split(",")[0].trim() || req.headers.get("x-real-ip") || "local";
}

/**
 * POST /api/ask {question, place, first?, since?} → an AskAnswer.
 * Answers come only from Neritic's PortFacts for one pilot port; see src/pilot/ask/engine.ts.
 */
export async function POST(req: Request) {
  let body: Partial<AskRequest>;
  try {
    body = (await req.json()) as Partial<AskRequest>;
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400, headers: NO_STORE });
  }
  const question = typeof body.question === "string" ? body.question : "";
  if (!question.trim()) return NextResponse.json({ error: "question is required" }, { status: 400, headers: NO_STORE });
  const key = clientKey(req);
  if (!allow("web", key)) return NextResponse.json({ error: "Too many questions; please wait a few minutes." }, { status: 429, headers: { ...NO_STORE, "Retry-After": "600" } });
  const since = body.since && typeof body.since === "object" && !Array.isArray(body.since) ? (body.since as SeenSnapshot) : null;
  const answer = await serveAsk(
    { question, place: typeof body.place === "string" ? body.place : null, first: body.first === true, since, channel: "web" },
    { userKey: key, channel: "web" },
  );
  return NextResponse.json(answer, { headers: NO_STORE });
}
