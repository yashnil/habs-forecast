import { TripsPage } from "@/nx/trips/ui/TripsPage";

export const metadata = { title: "Your trips" };

/**
 * Your trips (v15): the history (/coast/trips) and one trip's workspace (/coast/trips?id=…). One static page: every
 * trip lives in this browser's storage, so the server sends the same skeleton for any address (offline, the service
 * worker answers them all with one kept copy) and the page reads the id and the trips after it mounts.
 */
export default function Trips() {
  return <TripsPage />;
}
