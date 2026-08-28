# Native three-player poker engine

This flat folder contains the C++20 implementation and its Windows build files.
It is a parallel backend for `three_player_engine.py`; the Python engine remains
the correctness reference and fallback.

Build from a normal PowerShell or Command Prompt:

```bat
"engine C\build.bat"
```

The build copies `poker_native_engine*.pyd` to the repository root.  The Python
compatibility module `three_player_native.py` exposes the same environment API.

Never replace the reference backend merely because the extension compiles.
Run the differential and full regression tests first.
