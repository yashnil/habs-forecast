import { useRef } from "react";

/** Radio group drawn as a segmented control. Arrow keys move and select (roving tabindex). */
export function Segmented<T extends string>({ label, value, options, onChange, size = "md" }: {
  label: string;
  value: T;
  options: { id: T; label: string; short?: string; hint?: string }[];
  onChange: (v: T) => void;
  size?: "sm" | "md";
}) {
  const ref = useRef<HTMLDivElement>(null);
  const move = (dir: number) => {
    const i = options.findIndex((o) => o.id === value);
    const n = options[(i + dir + options.length) % options.length];
    onChange(n.id);
    requestAnimationFrame(() => ref.current?.querySelector<HTMLButtonElement>(`[data-id="${n.id}"]`)?.focus());
  };
  return (
    <div
      ref={ref}
      role="radiogroup"
      aria-label={label}
      className={`seg seg-${size}`}
      onKeyDown={(e) => {
        if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); move(1); }
        if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      }}
    >
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          role="radio"
          data-id={o.id}
          aria-checked={o.id === value}
          tabIndex={o.id === value ? 0 : -1}
          className="seg-opt"
          onClick={() => onChange(o.id)}
          title={o.hint}
        >
          {o.short ? <><span className="seg-long">{o.label}</span><span className="seg-short">{o.short}</span></> : o.label}
        </button>
      ))}
    </div>
  );
}
