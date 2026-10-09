import type { ReactNode } from "react";

export function Panel({ children, className = "", testid, labelledBy }: { children: ReactNode; className?: string; testid?: string; labelledBy?: string }) {
  return (
    <section data-testid={testid} aria-labelledby={labelledBy} className={`rounded-lg border border-hairline bg-surface ${className}`}>
      {children}
    </section>
  );
}

export function Eyebrow({ children }: { children: ReactNode }) {
  return <p className="text-[10.5px] font-semibold uppercase tracking-[0.08em] text-ink-3">{children}</p>;
}

export function SectionTitle({ id, children, aside }: { id?: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <h2 id={id} className="text-[13.5px] font-semibold tracking-tight text-ink">
        {children}
      </h2>
      {aside}
    </div>
  );
}

const TONE: Record<string, string> = {
  neutral: "border-hairline-strong bg-surface-2 text-ink-2",
  warning: "border-warning/45 bg-warning/[0.07] text-ink",
  serious: "border-serious/50 bg-serious/[0.08] text-ink",
  info: "border-accent/30 bg-accent/[0.06] text-ink-2",
};

export function Notice({ tone = "neutral", title, children, testid }: { tone?: keyof typeof TONE; title?: ReactNode; children?: ReactNode; testid?: string }) {
  return (
    <div data-testid={testid} role={tone === "serious" ? "alert" : undefined} className={`rounded-md border px-3 py-2 text-[12px] leading-snug ${TONE[tone]}`}>
      {title && <p className="mb-0.5 font-semibold text-ink">{title}</p>}
      {children}
    </div>
  );
}

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  label,
  testidPrefix,
}: {
  options: { value: T; label: ReactNode; sub?: ReactNode; disabled?: boolean; title?: string }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  testidPrefix?: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="grid gap-0.5 rounded-[9px] bg-surface-3 p-[3px]" style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={String(o.value)}
            role="radio"
            aria-checked={active}
            disabled={o.disabled}
            title={o.title}
            data-testid={testidPrefix ? `${testidPrefix}-${o.value}` : undefined}
            onClick={() => onChange(o.value)}
            className={`rounded-[7px] px-1.5 py-1.5 text-left text-[12px] transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40 ${
              active ? "bg-surface text-ink shadow-[0_1px_2px_rgba(13,27,42,0.12)] ring-1 ring-hairline-strong" : "text-ink-2 hover:text-ink"
            }`}
          >
            <span className="block font-medium leading-tight">{o.label}</span>
            {o.sub && <span className="block text-[10.5px] leading-tight text-ink-3">{o.sub}</span>}
          </button>
        );
      })}
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-surface-2 ${className}`} aria-hidden />;
}

export function SourceLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="text-accent underline-offset-2 hover:underline">
      {children}
    </a>
  );
}
