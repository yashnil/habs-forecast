"use client";

import "@/nx/ds/tokens.css";
import Link from "next/link";
import { ERROR_COPY } from "@/content/pages";
import { fontVariables } from "@/nx/ds/fonts";
import s from "@/nx/pages/global-error.module.css";

/**
 * The root layout itself failed: nothing of the app is available, so this page draws its own document in the design
 * system's tokens and type (src/nx/ds), says what happened and offers the two ways on.
 */
export default function GlobalError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <html lang="en" className={fontVariables} data-chart-mode="day">
      <body className={s.body}>
        <main className={s.page} data-testid="nx-global-error">
          <p className={s.brand}>Neritic</p>
          <h1 className={s.title}>{ERROR_COPY.errorTitle}</h1>
          <p className={s.lede}>{ERROR_COPY.errorLede}</p>
          <p className={s.actions}>
            {/* design-lint-allow raw-button: the design system's Button is not loaded when the root layout failed */}
            <button type="button" className={s.retry} onClick={reset}>
              {ERROR_COPY.retry}
            </button>
            <Link href="/" className={s.link}>
              Neritic home
            </Link>
          </p>
        </main>
      </body>
    </html>
  );
}
