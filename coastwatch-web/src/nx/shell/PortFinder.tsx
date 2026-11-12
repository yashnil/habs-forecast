"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Dialog } from "@/nx/ds/Overlay";
import { PortFinder as Finder } from "@/nx/ds/PortFinder";
import type { FinderData } from "@/nx/home/finder.server";
import { OPEN_PORT_FINDER as OPEN } from "./finder";
import { choosePort, useCurrentPort } from "./port";
import { hrefForPort } from "./redirects";

export { openPortFinder } from "./finder";


/**
 * The bar's port finder: the ds PortFinder (type, or read down the coast by CDFW port area) in a ds Dialog.
 * Choosing saves the port on this device and opens the same page there (from `/`, Today).
 */
export function PortFinder({ finder }: { finder: FinderData }) {
  const [open, setOpen] = useState(false);
  const router = useRouter();
  const pathname = usePathname();
  const { key } = useCurrentPort();
  useEffect(() => {
    const on = () => setOpen(true);
    window.addEventListener(OPEN, on);
    return () => window.removeEventListener(OPEN, on);
  }, []);
  const current = key ? Number(key.replace(/^port-/, "")) : undefined;
  return (
    <Dialog title="Find your port" open={open} onOpenChange={setOpen}>
      <div data-testid="nx-port-finder">
        {finder.ports.length ? (
          <Finder
            ports={finder.ports}
            regions={finder.regions}
            current={current}
            autoFocus
            label="Find your port"
            onSelect={(p) => {
              const k = finder.keys[p.code];
              if (!k) return;
              // the reader picked it: the first such choice also becomes the home port (shell/port.tsx)
              choosePort(k);
              setOpen(false);
              router.push(hrefForPort(pathname, k));
            }}
          />
        ) : (
          <p>The port list could not be read just now. Try again in a minute.</p>
        )}
      </div>
    </Dialog>
  );
}
