#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"
if [ ! -f "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import sys" 2>/dev/null; then
    echo "Setting up virtual environment..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -r "$DIR/requirements-prod.txt" -q
fi
exec "$VENV/bin/python" "$DIR/main.py"
