/**
 * The closing honesty line of an answer that rests on unverified official records ("Transcribed by
 * Neritic from CDFW and not yet verified by a person; the agency page is the authority."). A phone's
 * reply leaves it out and says "unverified" in its source tag instead (sms.ts).
 */
export const UNVERIFIED_LINE = /^Transcribed by Neritic\b/;
