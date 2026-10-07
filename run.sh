#!/bin/bash
# ZEKO VRE — Raspberry Pi 5 startup script.
# Activates the virtual environment and launches the engine.

# Determine virtual environment (.venv preferred, fallback to .build2r)
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
elif [ -f ".build2r/bin/activate" ]; then
    source ".build2r/bin/activate"
else
    echo "⚠️  Virtual environment not found (.venv or .build2r)."
    echo "Creating .venv..."
    python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
fi

# Ensure Audio access over SSH (PipeWire / PulseAudio session bus)
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export PYTHONUTF8=1

python main.py

