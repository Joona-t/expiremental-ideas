#!/usr/bin/env bash
# Guards against paid-LLM-API code creeping back into this repo (CLAUDE.md
# rule #10, BUG-001). Run manually with `scripts/check-no-paid-api.sh`, or
# install as a pre-push hook:
#
#   ln -sf ../../scripts/check-no-paid-api.sh .git/hooks/pre-push
#
# Exits non-zero (blocking the push) if a paid-API pattern is found in
# tracked, non-vendored source.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PATTERN='anthropic\.Anthropic\(|api\.anthropic\.com|ANTHROPIC_API_KEY|OPENAI_API_KEY'

# Scan tracked files only, so vendored/untracked third-party code never
# trips this. Exclude this guard script itself and the local-CLI provider,
# both of which reference the var *names* defensively (to strip them from
# subprocess env), never to read/use a key.
HITS=$(git grep -niE "$PATTERN" -- \
  ':!scripts/check-no-paid-api.sh' \
  ':!src/blitz_swarm/providers/local_cli.py' \
  ':!tests/test_local_cli_provider.py' \
  ':!README.md' \
  ':!BUGS_AND_ITERATIONS.md' \
  2>/dev/null || true)

if [ -n "$HITS" ]; then
  echo "check-no-paid-api: paid-API pattern found — this repo is local-CLI/mock only (rule #10):" >&2
  echo "$HITS" >&2
  exit 1
fi

echo "check-no-paid-api: clean"
