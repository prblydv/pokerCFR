# Native heads-up Hold'em engine

This directory is deliberately separate from `engine C`. It implements the
two-player integer-chip engine used by the heads-up migration and does not
replace or modify the existing three-player native extension.

Build on Windows:

```bat
"engine C HU\build.bat"
```

The script builds `heads_up_native_engine` and copies the extension to the
repository root. `heads_up_native.py` is the public compatibility facade.

The engine has two action interfaces:

- `step_exact(state, kind, raise_to=...)` applies exact poker-room actions.
- `step(state, slot)` applies one of the ten finite policy action templates.

The engine always keeps exact chip amounts. The finite template layer only
controls CFR branching; arbitrary observed raises are never rounded inside the
poker state.

The extension also exposes a blocker-aware batched all-in evaluator used by
the GUI resolver's independent safety pass. It samples weighted opponent
ranges and future boards entirely in C++, returning EV, standard error,
confidence bounds, call rate, and called equity without per-runout Python
crossings.

The stable policy slots are:

```text
fold, check, call, minimum raise, 1/3 pot, 1/2 pot, 3/4 pot,
pot, 1.5 pot, all-in
```

Illegal and effect-duplicate slots are masked. Named sizes are never silently
clamped to all-in; only the dedicated all-in slot may represent a short raise.

The native information-state encoder matches `heads_up_models.py`. With the
default 128 history events it emits 3,246 float32 features and requires exact
per-slot action descriptors for every live state. Encoding fails if a hand
exceeds the configured history capacity; increase `max_history` for deeper
stacks rather than silently merging distinct information states.

Verification:

```bat
python -m unittest -v test_heads_up_native_engine.py
```
