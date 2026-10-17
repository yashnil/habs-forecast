/** The first stop for keyboard and screen-reader readers: past the bar, to the page's own content. */
export function SkipLink({ to = "main", label = "Skip to main content" }: { to?: string; label?: string }) {
  return (
    <a
      href={`#${to}`}
      className="nx-focus sr-only z-[60] rounded-lg bg-n-surface px-3 py-2 text-[14px] font-semibold text-n-accent shadow-[var(--nx-shadow)] focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:flex focus:min-h-11 focus:items-center"
      data-testid="skip-link"
    >
      {label}
    </a>
  );
}
