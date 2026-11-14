/**
 * A small in-process LRU of promises (one per key): concurrent requests for the same upstream subset share one
 * fetch, and a failed fetch is dropped so the next request retries. Per server instance; the CDN and Next's fetch
 * cache sit in front of it.
 */
export class PromiseLru<T> {
  private m = new Map<string, Promise<T>>();
  constructor(private readonly cap: number) {}

  get(key: string, load: () => Promise<T>): Promise<T> {
    const hit = this.m.get(key);
    if (hit) {
      this.m.delete(key);
      this.m.set(key, hit);
      return hit;
    }
    const p = load();
    this.m.set(key, p);
    p.catch(() => { if (this.m.get(key) === p) this.m.delete(key); });
    while (this.m.size > this.cap) this.m.delete(this.m.keys().next().value as string);
    return p;
  }

  clear() {
    this.m.clear();
  }
}
