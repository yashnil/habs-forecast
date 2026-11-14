import { NotFoundBody } from "@/nx/pages/errors";

export const metadata = { title: "Not found" };

/** Not found, in the shell (src/nx/pages/errors.tsx). */
export default function NotFound() {
  return <NotFoundBody />;
}
