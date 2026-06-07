#!/usr/bin/env bash
# Legacy wrapper — use stop_saulie.sh for the full stack.
exec bash "$(dirname "$0")/stop_saulie.sh" "$@"
