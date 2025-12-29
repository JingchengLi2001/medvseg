#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the maintainable Python pipeline.
# Old long script is kept as: run_legacy.sh

python -m medvseg.pipelines.endoscopy_oneclick
