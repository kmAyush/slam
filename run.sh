#!/usr/bin/env bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export D2D=1

# Kill any leftover processes from a previous run
pkill -f "python.*slam.py" 2>/dev/null && sleep 0.5

exec uv run slam.py "$@"   # e.g. --top-down (default) or --diagonal
