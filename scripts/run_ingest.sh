#!/usr/bin/env bash
# Run one ingest. `set -e` stops at the first failing command and passes its exit code on.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m hp.ingest