"use client";

import { createContext, useContext, type ReactNode } from "react";

/** Whether the pilot's front door is on (NERITIC_PILOT=1), decided on the server (src/pilot/frontDoor.ts). */
const Ctx = createContext(false);

export function FrontDoorProvider({ pilot, children }: { pilot: boolean; children: ReactNode }) {
  return <Ctx.Provider value={pilot}>{children}</Ctx.Provider>;
}

export const usePilotFrontDoor = () => useContext(Ctx);
