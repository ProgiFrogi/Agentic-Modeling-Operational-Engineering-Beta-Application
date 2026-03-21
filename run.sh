#!/usr/bin/env bash
# Запуск из корня репозитория без ручного PYTHONPATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT"
cd "$ROOT"
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  exec "${VIRTUAL_ENV}/bin/python" cli.py "$@"
fi
if [[ -x "${ROOT}/venv/bin/python" ]]; then
  exec "${ROOT}/venv/bin/python" cli.py "$@"
fi
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  exec "${ROOT}/.venv/bin/python" cli.py "$@"
fi
exec python3 cli.py "$@"
