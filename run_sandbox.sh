#!/usr/bin/env bash
# Run an evaloop command inside the locked-down sandbox container.
#   ./run_sandbox.sh --help
#   ./run_sandbox.sh <evaloop args...>
#   ./run_sandbox.sh --python scripts/<script>.py [args...]   (offline, no network)
# Requires GEMINI_API_KEY in the environment (or in a .env file, gitignored)
# for evaloop runs.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

# --python: run a repo script (e.g. failure diagnostics) with no network at all.
MODE_FLAGS=(-e GEMINI_API_KEY)
if [[ "${1:-}" == "--python" ]]; then
  shift
  MODE_FLAGS=(--network none --entrypoint python -w /work)
else
  : "${GEMINI_API_KEY:?Set GEMINI_API_KEY (export it or put it in .env)}"
fi

mkdir -p results cache logs

# Git Bash on Windows rewrites /paths in arguments; disable that.
export MSYS_NO_PATHCONV=1

# Note: the container needs outbound network to reach the Gemini API, so
# generated code could also reach the network. Everything else is locked down:
# no capabilities, no privilege escalation, read-only root filesystem,
# memory/CPU/process limits, and only results/, cache/ and logs/ are writable.
TTY_FLAGS=(); [[ -t 0 ]] && TTY_FLAGS=(-it)

exec docker run --rm "${TTY_FLAGS[@]}" \
  --name "evaloop-$(date +%s)" \
  --user 1000:1000 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /tmp:rw,size=512m \
  --memory 4g --cpus 4 --pids-limit 512 \
  -v "$PWD/evaloop:/opt/evaloop/evaloop:ro" \
  -v "$PWD/scripts:/work/scripts:ro" \
  -v "$PWD/results:/work/results" \
  -v "$PWD/cache:/work/cache" \
  -v "$PWD/logs:/work/logs" \
  "${MODE_FLAGS[@]}" \
  evaloop-sandbox "$@"
