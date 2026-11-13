"use client";

import { useEffect } from "react";
import { ChartRegulation } from "@/nx/chart/regulation";
import { ChartScene } from "@/nx/chart/scene";
import { Button } from "@/nx/ds/Button";
import { PageLayout } from "@/nx/ds/layout";
import { Ticket, TicketMain } from "@/nx/ds/stage/Ticket";
import { openPortFinder } from "@/nx/shell/finder";
import { savePort, useCurrentPort, usePorts } from "@/nx/shell/port";
import { COAST_VIEW } from "@/nx/shell/views";
import type { Initial } from "./data";
import { TodayView } from "./TodayView";
import s from "./today.module.css";

/** /today: the address's port (a deep link wins, and becomes the reader's port), else the saved one. */
export function TodayPage({ initial }: { initial: Initial }) {
  const current = useCurrentPort();
  const ports = usePorts();
  // the port the server rendered for wins until the address names another: a saved copy of this page
  // answered at `/` offline (public/sw.js) hydrates the HTML it was given
  const seeded = !current.fromUrl && initial ? (ports.find((p) => p.key === initial.key) ?? null) : null;
  const port = seeded ?? current.port;
  const { key, fromUrl } = current;
  const pending = !seeded && current.pending;
  useEffect(() => {
    // the address named this port (a link, a tab): it is the port last opened, never a new home port
    if (fromUrl && key) savePort(key, { chosen: false });
  }, [key, fromUrl]);
  if (port) return <TodayView port={{ key: port.key, name: port.name, lon: port.lon, lat: port.lat }} initial={initial} />;
  if (pending) return <section className={s.today} data-testid="nx-today" />;
  return (
    <section className={s.today} data-testid="nx-today">
      <PageLayout mode="stage" phoneMap={0.27} />
      <ChartScene scene={{ id: "today:coast", view: COAST_VIEW }} />
      <ChartRegulation spec={{ zones: { highlight: null } }} />
      <Ticket at={null} width={404} labelledBy="nx-today-choose">
        <TicketMain>
          <h1 id="nx-today-choose" className={s.chooserTitle}>
            Today at your port
          </h1>
          <p className={s.chooserLead}>Choose your port to see its season status, the latest tests and the next decision date.</p>
          <p className={s.emptyAction}>
            <Button variant="primary" onClick={openPortFinder}>
              Find your port
            </Button>
          </p>
        </TicketMain>
      </Ticket>
    </section>
  );
}
