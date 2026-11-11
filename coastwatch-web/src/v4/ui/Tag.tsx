import type { ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

/**
 * A status tag (design.md §4.3): the status word with its shape on a filled tint, never colour alone and
 * never green; the calmest state is "None on record".
 *
 * <Tag level="delayed">Season delayed</Tag>
 * <Tag level="none" /> renders the words "None on record"; pair it with "not a promise it is open".
 */
export type TagLevel = "closed" | "delayed" | "restricted" | "watch" | "none";

const SHAPE: Record<TagLevel, IconName> = {
  closed: "noEntry",
  delayed: "hourglass",
  restricted: "alert",
  watch: "eye",
  none: "ring",
};
const WORD: Record<TagLevel, string> = {
  closed: "Closed",
  delayed: "Delayed",
  restricted: "Restricted",
  watch: "Watch",
  none: "None on record",
};

export function Tag({ level, children, testId, className = "" }: { level: TagLevel; children?: ReactNode; testId?: string; className?: string }) {
  return (
    <span className={`nx-tag ${className}`} data-level={level} data-testid={testId}>
      <Icon name={SHAPE[level]} className="h-3.5 w-3.5" />
      {children ?? WORD[level]}
    </span>
  );
}
