import { readiness } from "../readiness";
import { askDeterministic, type AskContext } from "./engine";
import { dailyCap, RateLimiter, SpendMeter } from "./limits";
import { askLive, createAnthropicClient, type MessagesClient } from "./live";
import { entryFor, hashKey, loggerFromEnv, type AskLogger } from "./logger";
import { loadPortToday, type PortTodayLoader } from "./source";
import type { AskAnswer, AskMeta, AskRequest } from "./types";

/**
 * Server-side entry point used by /api/ask and /api/sms: picks demo or live from readiness, applies
 * the per-user rate limit, and logs (no-op unless configured).
 */

type Env = Record<string, string | undefined>;

// per-instance state (see limits.ts on what that means for serverless)
const webLimiter = new RateLimiter(Number(process.env.NERITIC_ASK_WEB_LIMIT) || 30, 10 * 60_000); // 30 questions / 10 min per IP or account
const smsLimiter = new RateLimiter(10, 60 * 60_000); // 10 texts / hour per phone number
let meter: SpendMeter | null = null;
let meterCap = -1;
let client: MessagesClient | null = null;

export function now(env: Env = process.env): Date {
  const fixed = env.NERITIC_ASK_NOW;
  return fixed && Number.isFinite(Date.parse(fixed)) ? new Date(fixed) : new Date();
}

export function meta(env: Env = process.env, dataset: AskMeta["dataset"] = "fixture"): AskMeta {
  return { readiness: { ask: readiness("ask", env), sms: readiness("sms", env) }, logging: loggerFromEnv(env).enabled, dataset };
}

export type ServeOptions = {
  env?: Env;
  userKey: string | null;
  channel: "web" | "sms";
  clientOverride?: MessagesClient;
  loggerOverride?: AskLogger;
  loader?: PortTodayLoader;
  at?: Date;
};

export function allow(channel: "web" | "sms", key: string): boolean {
  return (channel === "sms" ? smsLimiter : webLimiter).take(key);
}

export async function serveAsk(req: AskRequest, o: ServeOptions): Promise<AskAnswer> {
  const env = o.env ?? process.env;
  const at = o.at ?? now(env);
  const ctx: AskContext = { now: at, mode: "demo", loader: o.loader ?? loadPortToday };
  let answer: AskAnswer;
  if (readiness("ask", env) === "live") {
    const cap = dailyCap(env);
    if (!meter || meterCap !== cap) {
      meter = new SpendMeter(cap);
      meterCap = cap;
    }
    client = o.clientOverride ?? client ?? (await createAnthropicClient(env.ANTHROPIC_API_KEY!));
    answer = await askLive(req, { ...ctx, mode: "live" }, { client, meter });
  } else {
    answer = await askDeterministic(req, ctx);
  }
  const logger = o.loggerOverride ?? loggerFromEnv(env);
  if (logger.enabled) {
    const loaded = answer.place ? await ctx.loader!(answer.place.place_id, at) : null;
    const retrieved = loaded?.today.facts.map((f) => f.id) ?? [];
    await logger.log(entryFor(req.question, answer, retrieved, o.channel, await hashKey(o.userKey, env.NERITIC_ASK_LOG_SALT), at));
  }
  return answer;
}
