/**
 * Rate limits and the daily spend cap. In-memory, per server instance: on serverless hosting each
 * warm instance keeps its own counters, so the effective cap is (cap × instances). That is why the
 * owner should ALSO set a monthly spend limit on the Anthropic Console workspace (docs/coastwatch/pilot/ask.md).
 */

export class RateLimiter {
  private hits = new Map<string, number[]>();
  constructor(
    private readonly limit: number,
    private readonly windowMs: number,
  ) {}

  /** true if allowed (and counted) */
  take(key: string, now = Date.now()): boolean {
    const from = now - this.windowMs;
    const list = (this.hits.get(key) ?? []).filter((t) => t > from);
    if (list.length >= this.limit) {
      this.hits.set(key, list);
      return false;
    }
    list.push(now);
    this.hits.set(key, list);
    if (this.hits.size > 5000) for (const [k, v] of this.hits) if (!v.some((t) => t > from)) this.hits.delete(k);
    return true;
  }
}

/** USD per million tokens (input, output), from the claude-api model table (cached 2026-10-06). */
export const PRICES: Record<string, { in: number; out: number }> = {
  "claude-sonnet-5-5": { in: 2, out: 10 },
  "claude-haiku-5-5": { in: 0.1, out: 0.5 },
};

export function costUsd(model: string, inputTokens: number, outputTokens: number): number {
  const p = PRICES[model] ?? { in: 5, out: 25 }; // unknown model: assume an expensive one
  return (inputTokens * p.in + outputTokens * p.out) / 1_000_000;
}

export const DEFAULT_DAILY_CAP_USD = 1;

export function dailyCap(env: Record<string, string | undefined>): number {
  const v = Number(env.NERITIC_ASK_DAILY_USD_CAP);
  return Number.isFinite(v) && v >= 0 && env.NERITIC_ASK_DAILY_USD_CAP !== undefined && env.NERITIC_ASK_DAILY_USD_CAP !== "" ? v : DEFAULT_DAILY_CAP_USD;
}

export class SpendMeter {
  private day = "";
  private spent = 0;
  constructor(private readonly cap: number) {}
  private roll(now: Date) {
    const d = now.toISOString().slice(0, 10);
    if (d !== this.day) {
      this.day = d;
      this.spent = 0;
    }
  }
  /** Would spending `estimate` more stay within today's cap? */
  allows(estimate: number, now = new Date()): boolean {
    this.roll(now);
    return this.spent + estimate <= this.cap;
  }
  add(usd: number, now = new Date()) {
    this.roll(now);
    this.spent += usd;
  }
  spentToday(now = new Date()): number {
    this.roll(now);
    return this.spent;
  }
}
