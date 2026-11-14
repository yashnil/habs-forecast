import Link from "next/link";
import { ERROR_COPY } from "@/content/pages";
import { Button } from "@/nx/ds/Button";
import { InfoPage, infoStyles as s } from "./InfoPage";

/** Where a reader can go from a page that is not there or could not be drawn. */
const PLACES: Array<[string, string]> = [
  ["/today", "Today"],
  ["/coast", "My Coast"],
  ["/research", "Research"],
  ["/notices", "Official notices"],
  ["/about", "About"],
  ["/sources", "Data & sources"],
];

function Places() {
  return (
    <nav aria-label="Neritic's pages" className={s.links}>
      {PLACES.map(([href, label]) => (
        <Link key={href} href={href} className={s.link}>
          {label}
        </Link>
      ))}
    </nav>
  );
}

/** Not found (v15.1), in the shell: no page at this address. */
export function NotFoundBody() {
  return (
    <InfoPage id="not-found" testId="nx-not-found" kicker={ERROR_COPY.notFoundKicker} title={ERROR_COPY.notFoundTitle} lede={ERROR_COPY.notFoundLede}>
      <Places />
    </InfoPage>
  );
}

/** A page that threw while drawing (v15.1), in the shell: nothing partial is shown; try again or go elsewhere. */
export function ErrorBody({ retry }: { retry?: () => void }) {
  return (
    <InfoPage id="error" testId="nx-error" kicker={ERROR_COPY.errorKicker} title={ERROR_COPY.errorTitle} lede={ERROR_COPY.errorLede}>
      {retry ? (
        <p className={s.after}>
          <Button variant="primary" onClick={retry} glyph={null}>
            {ERROR_COPY.retry}
          </Button>
        </p>
      ) : null}
      <Places />
    </InfoPage>
  );
}
