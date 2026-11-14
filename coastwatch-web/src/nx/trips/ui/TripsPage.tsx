"use client";

import { MotionConfig } from "motion/react";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { Micro, PageLayout, Skeleton, Ticket, TicketMain, TicketStub } from "@/nx/ds";
import { boundsAround, ChartScene } from "@/nx/chart/scene";
import { useTrip, useTrips } from "../useTrip";
import { EmptyTrips } from "./EmptyTrips";
import { Workspace } from "./Workspace";
import { useRouter } from "next/navigation";
import { useCurrentPort, usePorts } from "@/nx/shell/port";
import { useTripsData } from "../useTripsData";
import { keepTripsOffline } from "@/nx/offline/trips";
import { Briefing } from "./Briefing";
import { HistoryList } from "./HistoryList";
import s from "./trips.module.css";

/**
 * /coast/trips: the history ⇄ one trip's workspace (?id=). Client-side only: the server (and the offline copy) render
 * the same skeleton for every address; the id and the trips are read after mount.
 */
export function TripsPage() {
  useEffect(() => {
    void keepTripsOffline();
  }, []);
  return (
    <MotionConfig reducedMotion="user">
      <article className={s.page} data-testid="nx-trips">
        <Suspense fallback={<Waiting />}>
          <Route />
        </Suspense>
      </article>
    </MotionConfig>
  );
}

function Route() {
  const sp = useSearchParams();
  const id = sp.get("id");
  if (id && sp.get("view") === "briefing") return <Briefing tripId={id} />;
  return id ? <OneTrip id={id} /> : <History />;
}

/** The same shape on the server and before storage is read: a ticket holding a placeholder. */
function Waiting() {
  return (
    <>
      <PageLayout mode="stage" paneKey="trips" phoneMap={0.3} />
      <Ticket at={null} width={440} lead={false} label="Your trips" className={s.ticket}>
        <TicketMain className={s.main}>
          <Skeleton label="Reading the trips kept on this device…" lines={5} />
        </TicketMain>
      </Ticket>
    </>
  );
}

function OneTrip({ id }: { id: string }) {
  const model = useTrip(id);
  // the placeholder only before this device's storage is first read; never again while the trip is open
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    if (model.ready) setSeen(true);
  }, [model.ready]);
  if (!model.ready && !seen) return <Waiting />;
  if (!model.trip)
    return (
      <>
        <PageLayout mode="stage" paneKey="trips" phoneMap={0.3} />
        <Ticket at={null} width={440} labelledBy="trips-missing" className={s.ticket}>
          <TicketMain className={s.main}>
            <Micro>Your trips</Micro>
            <h1 id="trips-missing" className={s.emptyTitle}>
              This trip is not on this device.
            </h1>
            <p className={s.why}>Trips are kept in the browser they were saved in. It may have been deleted, or saved elsewhere.</p>
            <a className={s.src} href="/coast/trips">
              All trips on this device
            </a>
          </TicketMain>
        </Ticket>
      </>
    );
  return <Workspace model={model} />;
}

function History() {
  const trips = useTrips();
  const data = useTripsData();
  const ports = usePorts();
  const router = useRouter();
  // the chart stands at the reader's port (the whole coast without one)
  const { port } = useCurrentPort();
  return (
    <>
      {/* the history is not about one place: the chart is a backdrop that covers the whole sheet (the reader's waters, or
          the coast), so no panel reserves a side and no neatline cuts it */}
      <ChartScene
        scene={{
          id: `trips:history:${port?.key ?? "coast"}`,
          view: { cover: port ? boundsAround([port.lon + 0.6, port.lat], 240, 140) : [[-126.5, 32.2], [-116.5, 42.2]], position: [0.5, 0.5] },
          focus: port ? { key: port.key, name: port.name, lon: port.lon, lat: port.lat } : null,
        }}
      />
      <PageLayout mode="stage" paneKey="trips" phoneMap={0.3} />
      {trips === null ? (
        <Waiting />
      ) : (
        <Ticket at={null} lead={false} width={trips.length ? 440 : 404} labelledBy="trips-title" className={s.ticket} data-testid="trips-history">
          <TicketMain className={s.main}>
            {trips.length ? (
              <>
                <header className={s.heading} data-nx-desk-header="">
                  <Micro>My Coast</Micro>
                  <h1 id="trips-title" className={s.historyTitle}>
                    Your trips
                  </h1>
                </header>
                <HistoryList data={data} now={new Date()} portName={(k) => ports.find((p) => p.key === k)?.name ?? null} onOpen={(id) => router.push(`/coast/trips?id=${encodeURIComponent(id)}`)} />
              </>
            ) : (
              <EmptyTrips />
            )}
          </TicketMain>
          {trips.length ? null : (
            <TicketStub className={s.stub}>
              <p className={s.local} data-testid="trips-local">
                Kept on this device only, never sent to Neritic.
              </p>
            </TicketStub>
          )}
        </Ticket>
      )}
    </>
  );
}
