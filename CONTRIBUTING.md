# Contributing

Thank you for helping improve pokerCFR.

## Development setup

1. Fork and clone the repository.
2. Create a Python 3.11+ virtual environment.
3. Install `requirements-heads-up.txt` for the primary HUNL path.
4. Create a focused branch and keep unrelated changes separate.
5. Run the relevant unit tests before opening a pull request.

The Python engine is the readable correctness reference. If native behavior changes, update the Python and C++ implementations together and run their equivalence tests.

## Pull-request checklist

- Explain the poker rule, CFR behavior, or interface being changed.
- Add or update tests for legal actions, chip conservation, terminal utility, encodings, and checkpoint compatibility as applicable.
- Report the exact test command and result.
- Keep heads-up and three-player schemas isolated.
- Label simulated or learned metrics accurately; do not describe them as guaranteed profit or exact GTO performance.
- Do not commit generated checkpoints, weights, hand histories, credentials, private keys, compiled binaries, or large artifact directories.

## Style

- Prefer explicit invariants and deterministic seeds in correctness tests.
- Use integer chips in engine transitions.
- Keep exact room actions separate from finite policy slots.
- Preserve backward-compatible metadata checks rather than silently accepting incompatible checkpoints.
