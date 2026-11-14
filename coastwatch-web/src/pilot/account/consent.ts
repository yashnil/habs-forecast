/**
 * The exact words a person agrees to. They are shown verbatim on the sign-up form, stored verbatim
 * in the consent record (consent_log.text_shown / alert_subscriptions.sms_consent_text), and quoted
 * on /sms-terms. Changing a word here changes LEGAL_VERSION.
 */
export const LEGAL_VERSION = "2026-11-14";
export const LEGAL_EFFECTIVE = "November 14, 2026";
export const PROGRAM_NAME = "Neritic Alerts";

export const AGE_TEXT = "I am 13 or older, and I have read the Terms of Use and the Privacy Policy.";

export const SMS_CONSENT_TEXT =
  `Yes, text me ${PROGRAM_NAME} at the number above about the ports and changes I choose (official status changes, new toxin test results, whale-entanglement actions, bloom-risk jumps). ` +
  "Message frequency varies, up to 5 msgs/week. Msg & data rates may apply. Reply STOP to cancel, HELP for help. " +
  "Consent is not a condition of using Neritic or of any purchase. See the SMS Terms and Privacy Policy.";

export const PUSH_CONSENT_TEXT = "Send alerts for my saved ports to this device as notifications. You can turn them off here or in your browser settings.";

/** E.164 for US numbers typed loosely ("(831) 555-0100", "831.555.0100", "+1 831 555 0100"); null if not a US number. */
export function toE164US(raw: string): string | null {
  const d = raw.replace(/[^\d+]/g, "");
  const digits = d.replace(/\D/g, "");
  if (digits.length === 10 && /^[2-9]/.test(digits)) return `+1${digits}`;
  if (digits.length === 11 && digits.startsWith("1") && /^[2-9]/.test(digits.slice(1))) return `+${digits}`;
  return null;
}
