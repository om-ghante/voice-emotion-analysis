#!/bin/bash
set -e

PORT="${PORT:-10000}"

exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
