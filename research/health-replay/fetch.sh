#!/bin/bash
# Download each data-refresh run's manifest.json and health.json (from the run's
# "coastwatch-data" artifact) into $1/runs/<run id>/. Read-only; publishes nothing.
set -euo pipefail
D=${1:?output directory}
mkdir -p "$D/runs" "$D/tmp"
gh run list --workflow coastwatch-data.yml --limit 200 --json databaseId,createdAt,conclusion,event,headBranch,headSha > "$D/runs.json"
gh api "repos/:owner/:repo/actions/artifacts?per_page=100&name=coastwatch-data" --paginate -q '.artifacts[]|[.id,.workflow_run.id]|@tsv' |
while IFS=$'\t' read -r aid rid; do
  [ -f "$D/runs/$rid/manifest.json" ] && continue
  mkdir -p "$D/runs/$rid"
  gh api "repos/:owner/:repo/actions/artifacts/$aid/zip" > "$D/tmp/a.zip"
  unzip -o -q -j "$D/tmp/a.zip" 'v1/manifest.json' 'v1/health.json' -d "$D/runs/$rid" || true
  rm -f "$D/tmp/a.zip"
done
