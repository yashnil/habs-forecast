import type { AskAnswer } from "./types";

/**
 * Pluggable question log. Default: no-op (nothing is stored). When configured, one row per
 * question goes to a Supabase table through PostgREST (fetch only, no SDK). Retention: 90 days,
 * enforced by a scheduled delete (SQL in docs/coastwatch/pilot/ask.md). The user key is a salted
 * SHA-256 hash of the IP / account id / phone number, never the raw value.
 */

export type AskLogEntry = {
  ts: string;
  channel: "web" | "sms";
  mode: AskAnswer["mode"];
  model: string | null;
  place_id: string | null;
  question: string;
  intent: string;
  outcome: string;
  retrieved_fact_ids: string[];
  cited_fact_ids: string[];
  answer: string;
  guard_codes: string[];
  user_key: string | null;
};

export interface AskLogger {
  readonly enabled: boolean;
  log(entry: AskLogEntry): Promise<void>;
}

export const noopLogger: AskLogger = { enabled: false, log: async () => {} };

export class SupabaseAskLogger implements AskLogger {
  readonly enabled = true;
  constructor(
    private readonly url: string,
    private readonly serviceKey: string,
    private readonly table = "ask_log",
    private readonly fetchImpl: typeof fetch = fetch,
  ) {}
  async log(entry: AskLogEntry): Promise<void> {
    try {
      await this.fetchImpl(`${this.url.replace(/\/$/, "")}/rest/v1/${this.table}`, {
        method: "POST",
        headers: { apikey: this.serviceKey, Authorization: `Bearer ${this.serviceKey}`, "Content-Type": "application/json", Prefer: "return=minimal" },
        body: JSON.stringify(entry),
      });
    } catch {
      // logging never breaks an answer
    }
  }
}

/** NERITIC_ASK_LOG=supabase + SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) + SUPABASE_SERVICE_ROLE_KEY. */
export function loggerFromEnv(env: Record<string, string | undefined>): AskLogger {
  const url = env.SUPABASE_URL ?? env.NEXT_PUBLIC_SUPABASE_URL;
  if (env.NERITIC_ASK_LOG === "supabase" && url && env.SUPABASE_SERVICE_ROLE_KEY) return new SupabaseAskLogger(url, env.SUPABASE_SERVICE_ROLE_KEY, env.NERITIC_ASK_LOG_TABLE || "ask_log");
  return noopLogger;
}

export async function hashKey(raw: string | null, salt: string | undefined): Promise<string | null> {
  if (!raw) return null;
  const data = new TextEncoder().encode(`${salt ?? "neritic-ask"}:${raw}`);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", data));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 32);
}

export function entryFor(question: string, a: AskAnswer, retrieved: string[], channel: "web" | "sms", userKey: string | null, now: Date): AskLogEntry {
  return {
    ts: now.toISOString(),
    channel,
    mode: a.mode,
    model: a.model,
    place_id: a.place?.place_id ?? null,
    question: question.slice(0, 400),
    intent: a.intent,
    outcome: a.outcome,
    retrieved_fact_ids: retrieved,
    cited_fact_ids: a.citations.map((c) => c.fact_id),
    answer: [...a.preface, ...a.lines.map((l) => `${l.text} [${l.cites.join(",")}]`)].join("\n"),
    guard_codes: a.guard.codes,
    user_key: userKey,
  };
}
