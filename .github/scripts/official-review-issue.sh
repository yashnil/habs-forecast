#!/usr/bin/env bash
# Open, update or close the official-notices review issue from the watcher's report.
#
# One issue (found by its exact, legacy title: a new title would open a duplicate). A comment
# is added only when the watcher says the set of conditions changed (NOTIFY=true: something
# appeared, changed again or resolved), and never twice for the same set: the report carries
# a marker with a hash of the open conditions, and the last marker already on the issue wins
# when it matches (this also covers a run whose state file was not published). When nothing is
# left for a person, the issue is closed with the final report.
#
# Environment: GH_TOKEN, GITHUB_REPOSITORY, NOTIFY (true|false), CONDITIONS (count), MARKER,
# REPORT (markdown file). Nothing here edits the registry or marks anything reviewed.
set -euo pipefail

title="Official notices: sources changed since last review"
tag="neritic-official-watch:"
n=$(gh issue list --repo "$GITHUB_REPOSITORY" --state open --search "\"$title\" in:title" --json number --jq '.[0].number')

if [ "${NOTIFY:-false}" != "true" ]; then
  echo "Conditions unchanged since the last run (${CONDITIONS:-0} open); not commenting on ${n:+#$n}${n:-any issue}."
  exit 0
fi

if [ "${CONDITIONS:-0}" -gt 0 ]; then
  if [ -z "$n" ]; then
    gh issue create --repo "$GITHUB_REPOSITORY" --title "$title" --body-file "$REPORT"
    exit 0
  fi
  # the newest marker already on the issue: the issue body, then every comment in order
  last=$( { gh api "repos/$GITHUB_REPOSITORY/issues/$n" --jq '.body // ""';
            gh api --paginate "repos/$GITHUB_REPOSITORY/issues/$n/comments?per_page=100" --jq '.[].body // ""'; } \
          | grep -o "$tag [0-9a-f]*" | tail -1 | awk '{print $2}' || true)
  if [ -n "${MARKER:-}" ] && [ "$last" = "$MARKER" ]; then
    echo "The same conditions ($MARKER) are already the latest report on #$n; not commenting again."
    exit 0
  fi
  gh issue comment "$n" --repo "$GITHUB_REPOSITORY" --body-file "$REPORT"
elif [ -n "$n" ]; then
  gh issue comment "$n" --repo "$GITHUB_REPOSITORY" --body-file "$REPORT"
  gh issue close "$n" --repo "$GITHUB_REPOSITORY"
else
  echo "Nothing open and nothing left for a person."
fi
