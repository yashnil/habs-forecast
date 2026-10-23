/** One short visual confirmation at a time (MotionRoot's Toaster draws it). Screen readers hear the control's own live text. */
const subs = new Set<(text: string) => void>();

export function toast(text: string) {
  subs.forEach((f) => f(text));
}

export function onToast(f: (text: string) => void): () => void {
  subs.add(f);
  return () => {
    subs.delete(f);
  };
}
