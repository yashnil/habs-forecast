/**
 * A page arrives: reading pages rise 10 px and fade in; map pages only fade (the canvas is never
 * transformed). Plain CSS, so it runs from the first server-rendered frame, without waiting for
 * JavaScript; reduced motion turns it off (globals.css).
 */
export function PageIn({ kind, children }: { kind: "app" | "page"; children: React.ReactNode }) {
  return <div className={`flex min-h-0 flex-1 flex-col ${kind === "page" ? "cw-page-in" : "cw-fade-in"}`}>{children}</div>;
}
