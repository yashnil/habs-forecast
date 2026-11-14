import { loadTrust } from "@/lib/data";
import { ACTION_LABEL, FISHERY_LABEL, currentRecords, officialVerification } from "@/lib/official";
import type { RegistryTrust } from "./RegistryNote";

/** Now, or the fixture clock (tests and fixture previews). */
export const serverNow = () => (process.env.CW_FIXED_NOW ? new Date(process.env.CW_FIXED_NOW) : new Date());

/**
 * What an informational page's registry line says, read from the published dataset on the server: the state of the
 * notice registry (anything but "verified" is said as not verified), the current notices by name, and whether the
 * data are the recorded fixture. Slim: the records' geometry never reaches the browser.
 */
export async function registryTrust(): Promise<RegistryTrust> {
  const t = await loadTrust();
  const v = officialVerification(t.official, t.status, serverNow());
  return {
    fixture: t.fixture,
    generatedAt: t.generatedAt,
    state: t.official ? v.state : "unavailable",
    reasons: v.reasons,
    records: t.official
      ? currentRecords(t.official).map((r) => ({ id: r.id, title: r.title, line: `${r.agency} · ${ACTION_LABEL[r.action]} · ${FISHERY_LABEL[r.fishery]}` }))
      : null,
  };
}
