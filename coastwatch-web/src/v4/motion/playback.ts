"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { prefersReducedMotion } from "./reduced";

export const PLAYBACK_MS = 900;

/**
 * The clock behind usePlayback, free of React so its rules are tested directly: it only ever moves between
 * whole frames. At the last frame it stops, unless `loop` is set.
 */
export function playbackClock({ count, intervalMs, loop, onStep, onStop }: { count: number; intervalMs: number; loop: boolean; onStep: (index: number) => void; onStop: () => void }) {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let index = 0;
  const stop = () => {
    if (timer != null) clearTimeout(timer);
    timer = null;
  };
  const tick = () => {
    const at = index + 1;
    if (at >= count) {
      if (!loop || count < 2) {
        stop();
        onStop();
        return;
      }
      index = 0;
    } else index = at;
    onStep(index);
    timer = setTimeout(tick, intervalMs);
  };
  return {
    start(from: number) {
      stop();
      index = from;
      if (count < 2) return void onStop();
      timer = setTimeout(tick, intervalMs);
    },
    stop,
    running: () => timer != null,
  };
}

/** Input that stops playback, unless it lands on a control marked `data-playback-control` (Play, Pause). */
export const PLAYBACK_STOP_EVENTS = ["pointerdown", "keydown", "wheel", "touchstart"] as const;
export function stopsPlayback(target: EventTarget | null, key?: string): boolean {
  const t = target as { closest?: (s: string) => unknown } | null;
  if (t?.closest?.("[data-playback-control]")) return false;
  // Space inside a [data-playback-space] region is that region's own play/pause key: stopping here as well would
  // pause first and let its handler start playback again. A real button there keeps Space for itself.
  if (key === " " && t?.closest?.("[data-playback-space]") && !t.closest?.('button:not([role="radio"])')) return false;
  return true;
}

/**
 * Steps through discrete frames (C-HARM days, satellite days), one per `intervalMs`. It never interpolates:
 * `frame` is always one of the frames you passed, so each label is that frame's own date and kind.
 * Any user input stops it. Put `data-playback-control` on your Play/Pause button so pressing it is not that input.
 * Reduced motion: `autoStart` is ignored and it never loops (a press of Play still steps, as a cut).
 *
 * Use for: the time rail's Play. Not for: anything that does not have real frames (nothing to play between).
 */
export function usePlayback<F>(frames: readonly F[], { intervalMs = PLAYBACK_MS, loop = false, autoStart = false, initial = 0 }: { intervalMs?: number; loop?: boolean; autoStart?: boolean; initial?: number } = {}) {
  const [index, setIndex] = useState(() => Math.max(0, Math.min(frames.length - 1, initial)));
  const [playing, setPlaying] = useState(false);
  const n = frames.length;
  const at = Math.max(0, Math.min(n - 1, index));
  const clock = useRef<ReturnType<typeof playbackClock> | null>(null);
  const atRef = useRef(at);
  useEffect(() => {
    atRef.current = at;
  }, [at]);

  const pause = useCallback(() => {
    clock.current?.stop();
    setPlaying(false);
  }, []);
  const play = useCallback(() => {
    clock.current?.stop();
    const c = playbackClock({ count: n, intervalMs, loop: loop && !prefersReducedMotion(), onStep: setIndex, onStop: () => setPlaying(false) });
    clock.current = c;
    // from the end, a press of Play starts again
    const from = atRef.current >= n - 1 ? 0 : atRef.current;
    if (from !== atRef.current) setIndex(from);
    c.start(from);
    setPlaying(c.running());
  }, [n, intervalMs, loop]);
  const seek = useCallback(
    (i: number) => {
      pause();
      setIndex(Math.max(0, Math.min(n - 1, i)));
    },
    [n, pause],
  );

  useEffect(() => {
    if (autoStart && !prefersReducedMotion()) play();
    return () => clock.current?.stop();
    // on mount, and when the set of frames changes length
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [n]);

  useEffect(() => {
    if (!playing) return;
    const stop = (e: Event) => {
      if (stopsPlayback(e.target, (e as KeyboardEvent).key)) pause();
    };
    const opts = { capture: true, passive: true } as const;
    for (const ev of PLAYBACK_STOP_EVENTS) window.addEventListener(ev, stop, opts);
    return () => {
      for (const ev of PLAYBACK_STOP_EVENTS) window.removeEventListener(ev, stop, opts);
    };
  }, [playing, pause]);

  return useMemo(() => ({ index: at, frame: n ? frames[at] : undefined, playing, play, pause, toggle: () => (playing ? pause() : play()), seek }), [at, n, frames, playing, play, pause, seek]);
}
