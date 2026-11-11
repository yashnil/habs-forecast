import Link from "next/link";
import type { ButtonHTMLAttributes, ComponentProps, ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

/**
 * The three buttons (design.md §4.3): primary is ink-filled, secondary surface-filled, tertiary is
 * underlined text. 44 px by default; a primary action is 56 px on phones (CSS), and `size="lg"` is 56 px
 * everywhere. Press and hover feedback come with them.
 *
 * <Button kind="primary" onClick={…}>Open today's briefing</Button>
 * <LinkButton kind="tertiary" href="/sources">Data status</LinkButton>
 * <LinkButton kind="secondary" external href="https://…">CDFW</LinkButton>  (a new tab, an external mark)
 */
export type ButtonKind = "primary" | "secondary" | "tertiary";
type Common = { kind?: ButtonKind; size?: "md" | "lg"; icon?: IconName; children: ReactNode; testId?: string };

const cls = (kind: ButtonKind, size: "md" | "lg", extra = "") => `nx-btn nx-btn-${kind} ${size === "lg" ? "nx-btn-lg" : ""} nx-press nx-focus ${extra}`.trim();

export function Button({ kind = "secondary", size = "md", icon, children, testId, className = "", ...rest }: Common & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button type="button" {...rest} className={cls(kind, size, className)} data-testid={testId}>
      {icon && <Icon name={icon} className="h-[18px] w-[18px]" />}
      {children}
    </button>
  );
}

export function LinkButton({ kind = "secondary", size = "md", icon, children, testId, className = "", href, external = false, ...rest }: Common & Omit<ComponentProps<typeof Link>, "href"> & { href: string; external?: boolean }) {
  const inner = (
    <>
      {icon && <Icon name={icon} className="h-[18px] w-[18px]" />}
      {children}
      {external && <Icon name="external" className="h-3.5 w-3.5" />}
    </>
  );
  return external ? (
    <a href={href} target="_blank" rel="noreferrer" className={cls(kind, size, className)} data-testid={testId}>
      {inner}
    </a>
  ) : (
    <Link href={href} {...rest} className={cls(kind, size, className)} data-testid={testId}>
      {inner}
    </Link>
  );
}
