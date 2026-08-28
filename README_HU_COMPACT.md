# Compact hidden-384 HU campaign

This package starts a new heads-up Deep CFR campaign. It never resumes or
rewrites legacy checkpoints or reservoirs.

- Encoder schema: `hu_compact_information_state_v1_full_history`
- Physical replay/inference width: 782 float values
- Logical width: `40 + 7 * public_history_length`
- History capacity: all 106 public events reachable at a live decision in the
  current 100BB engine; overflow is an error, never silent truncation
- Cards: seven exact card IDs (`1..52`) with zero padding
- Network: structured hidden-384 card/history/action model with fixed
  sinusoidal history positions and an exact 1,326-combination range head
- Engine/actions: unchanged integer-chip HU engine and ten action slots
- CFR/regret/all-in behavior: unchanged in this package

The new `heads_up_compact_training.ipynb` retains the established random,
calling-station, TAG, range, BB confidence-interval, architecture and strategy
plots. It also evaluates the live compact policy against the frozen
725/950/1025 top-three ensemble. Each side uses its own compatible encoder.

On Vast.ai:

```bash
bash setup_vast_compact_v6.sh
```

Then open `heads_up_compact_training.ipynb`. Training artifacts are written
only under `artifacts/heads_up_compact_v6_hidden384`.

The three frozen policy files are read-only evaluation inputs. This archive
contains no training checkpoints or replay reservoirs.
