# Three-player neural CFR poker bot

This is a separate three-handed Hold'em implementation. It does **not** import
the original heads-up `engine.py`, because that engine has invalid terminal
payoffs, showdown ranking, and heads-up action order.

## What is included

- `three_player_engine.py`: rotating button, correct three-handed action order,
  strict legal actions, all-ins, side pots, ties, zero-sum settlement, and a
  dependency-free seven-card evaluator.
- `engine C`: the flat C++20 source/build folder for the packed native engine.
  `three_player_native.py` provides its Python-compatible facade; the production
  notebook uses this backend while retaining `three_player_engine.py` as the
  differential-test reference and fallback.
- `three_player_models.py`: a hero-visible information-state encoder, three
  advantage networks, and three average-policy networks.
- `three_player_cfr.py`: reach-corrected external-sampling neural CFR,
  batched policy inference, variable-stack three-handed/heads-up hand roots,
  continuation sampling, phase metrics, and resumable checkpoints.
- `three_player_tournament.py`: continuing stacks, elimination, heads-up play,
  and winner-take-all tournament results across hands.
- `three_player_production.py`: fixed held-out opponent suites, paired confidence
  estimates, policy-only historical snapshots, champion promotion, versioned
  checkpoints, emergency recovery, and the nine-panel training dashboard.
- `three_player_analysis.py`: legally replayed decision scenarios, exact 1,326
  combination analysis, 169-hand action-frequency maps, postflop blockers,
  checkpoint deltas, and next-card sensitivity plots.
- `train_three_player.py`: fixed-stack and isolated tournament command-line runs.
- `three_player_training.ipynb`: the production control plane. Its default
  profile targets 10,000 CUDA iterations and resumes automatically.
- `play_three_player_gui.py`: a persistent-stack human-versus-two-bot
  tournament GUI.
- `real_time_search.py`: a latency-bounded, public-information-safe re-solver
  using paired card determinizations, root regret matching, blueprint
  continuation rollouts, and bounded policy improvement.
- `test_three_player_engine.py`, `test_tournament_engine.py`,
  `test_three_player_training.py`, `test_three_player_tournament.py`, and
  `test_three_player_analysis.py`: engine, tournament, CUDA, campaign, and
  range-analysis regression tests.

## Setup

```powershell
python -m pip install -r requirements-three-player.txt
& '.\engine C\build.bat'
python -m unittest -v test_three_player_engine.py test_tournament_engine.py test_three_player_training.py test_tournament_training.py test_three_player_tournament.py test_three_player_analysis.py
jupyter lab three_player_training.ipynb
```

To play against downloaded policy 8100 with one scripted TAG seat, run
`run_three_player_gui.bat`. The high-think laptop preset uses a 7-second
budget and a 150,000-rollout safety cap per policy decision. Search runs on a
worker thread so the table stays responsive. Override it with, for example,
`run_three_player_gui.bat --search-ms 5000`, or pass
`--no-search` to compare against the raw checkpoint.

The real-time resolver borrows the blueprint-plus-subgame-search structure used
by systems such as Pluribus. It is a resource-bounded approximation, not the
full Pluribus algorithm or a claim of equivalent strength.

The notebook imports the modules above; it does not hide a second copy of the
engine or trainer inside notebook cells. Its default `tournament_production`
profile writes under `artifacts/three_player_tournament_production`. It cannot
resume the old fixed-stack reservoirs under `artifacts/three_player_production`.
To validate the tournament plumbing in another isolated directory, start
Jupyter from a shell containing:

```powershell
$env:POKER_TRAINING_PROFILE = "tournament_validation"
jupyter lab three_player_training.ipynb
```

The selectable profiles are:

- `tournament_production` (default): 600 total chips, variable three-handed and
  heads-up roots, continuation roots, full-tournament evaluation, and a one-time
  warm start from `policy_00000900.pt` when the new artifact directory is empty.
- `tournament_validation`: a small random-initialized test using its own
  artifact directory.
- `production` and `validation`: the original fixed-stack workflows, preserved
  for compatibility.

Increase an existing campaign's absolute target without changing its persisted
training settings:

```powershell
$env:POKER_TARGET_ITERATION = "20000"
jupyter lab three_player_training.ipynb
```

Full preflop/postflop hand charts refresh after every completed validation
evaluation by default. Change that independently of evaluation frequency:

```powershell
$env:POKER_RANGE_EVERY_EVALS = "2"
jupyter lab three_player_training.ipynb
```

Cell 6 prints the most recent completed validation composite between evaluation
boundaries and a countdown to the next one. A missing composite on an ordinary
training row means “not evaluated this iteration,” not a failed or NaN model.

## Quick command-line run

The legacy fixed-stack command remains unchanged:

```powershell
python train_three_player.py --iterations 10 --traversals 1 --stack 40 --device cuda
```

The default 40-chip stack is a quick 20-BB demonstration. For a 100-BB game:

```powershell
python train_three_player.py --iterations 1000 --traversals 8 --stack 200 --adv-steps 128 --policy-steps 64 --eval-every 25 --eval-games 1998 --device cuda
```

For variable tournament depths, elimination-aware states, and heads-up roots,
use `--tournament`. This command expands the state encoder and warm-starts its
three average-policy networks from the existing iteration-900 snapshot. The
source uses hidden width 256, three residual blocks, and history length 32, so
those architecture values must match:

```powershell
python train_three_player.py --tournament --iterations 1000 --traversals 8 --stack 200 --tournament-total-chips 600 --heads-up-root-fraction 0.25 --continuation-root-fraction 0.25 --hidden 256 --blocks 3 --adv-steps 128 --policy-steps 64 --eval-every 25 --eval-games 1998 --tournament-eval-games 1 --warm-start-policy artifacts/three_player_production/snapshots/policy_00000900.pt --checkpoint artifacts/three_player_tournament_cli/three_player_cfr.pt --metrics artifacts/three_player_tournament_cli/metrics.json --device cuda
```

Without explicit output paths, `--tournament` writes under the new
`artifacts/three_player_tournament` directory; it never selects the legacy
`artifacts/three_player_cfr.pt`. The defaults train on a 25% heads-up-root
mixture and draw 25% of roots from real prior-hand continuations when available.
Synthetic roots conserve the configured tournament chip total and deliberately
cover balanced, deep, and short stacks.

Start smaller and measure traversal time before committing to a large run. The
notebook requires CUDA and verifies that all six networks are on `cuda:0`.
Recursive traversal performs scalar inference and may not saturate a GPU;
batched replay fitting provides most of the acceleration. CUDA allocated,
reserved, and peak memory are plotted in the notebook.

Resume a fixed-stack full checkpoint with the same stack/blind configuration:

```powershell
python train_three_player.py --resume artifacts/three_player_cfr.pt --iterations 100 --stack 40 --device cuda
```

Resume the tournament command above without applying the warm start again:

```powershell
python train_three_player.py --resume artifacts/three_player_tournament_cli/three_player_cfr.pt --iterations 100 --stack 200 --device cuda --metrics artifacts/three_player_tournament_cli/metrics.json
```

The checkpoint itself restores whether tournament features and variable roots
are enabled. `--warm-start-policy` and `--resume` are intentionally mutually
exclusive. A policy warm start imports only the three average-policy networks;
advantage networks, optimizer memories, and reservoirs start fresh because the
new encoder and training distribution are different.

Each player-specific network is trained across all positions. The trainer
explicitly cycles every traverser through button, small blind, and big blind,
independently of ordinary dealer rotation.

Strict mode refits each advantage network after traversal and always performs
at least one complete reservoir pass; `--adv-steps` is a minimum. The optional
`--warm-start-advantages` mode is faster but is an explicitly looser neural-CFR
approximation.

## Tournament state and objective

Tournament snapshots append 15 features after the complete legacy state vector.
They include each seat's alive flag and starting-chip share, hero stack behind,
both effective stacks, players remaining, players still in the hand, a heads-up
flag, tournament chip scale, and shortest/largest live starting stacks. A zero
current stack during a hand still means all-in; only a zero starting stack means
the seat was eliminated before the hand. The engine skips eliminated seats and
uses correct button/small-blind/big-blind and action order when two players
remain. With one survivor, the tournament is over and no new hand is dealt.

Training mixes independently sampled chip-conserving hand roots with roots
produced by self-play continuation hands. CFR utility is still the chip result
of that hand. It does **not** recurse through thousands of hands as one solved
extensive-form tournament and does not backpropagate the final +2/-1/-1 result.
That final winner-take-all reward is measured by complete persistent-stack
tournament rollouts during the tournament production profile. Metrics include
win rate, bust rate, mean reward, mean finishing position, and hands/actions to
completion. This separation gives useful tournament evaluation without making
a false claim that the full multi-hand tournament was solved.

Blinds currently remain fixed at 1/2. Blind-level increases and ICM/payout
structures other than winner-take-all are not implemented. If blind levels are
added later, the level and blind sizes must also become policy inputs.

## Production evaluation and graphs

Every configured evaluation interval, the campaign uses the same validation
deals against uniform-random, calling-station, and card-aware tight-aggressive
opponents. It records clustered confidence intervals, per-position EV, paired
change versus the frozen initial policy, and EV against recent historical
snapshots. Tournament profiles additionally play a deliberately small set of
complete continuing-stack tournaments against the configured tournament
opponents and save their per-tournament rows beside the hand-EV CSVs. A separate
seed and larger hand count are reserved for the final holdout cell. Network
loss, sampled regret, rollout fallbacks, phase timing,
reservoir size, importance-weight ESS, and VRAM remain health diagnostics—not
playing-strength claims.

The 13x13 charts average every physical suit combination and all three policy
networks. They show fold, call, or raise *policy frequency* in an exact named
state; they do not show hand equity or chance of winning. Postflop charts remove
blocked cards, and each scenario is produced by a legal engine replay rather
than manually patched state fields.

The encoder retains the latest 32 public actions plus pending actors,
raise-rights, and current betting state. Exceptionally long betting lines are
therefore an explicit imperfect-recall abstraction. Multiplayer inverse-reach
weights are capped for variance control; the notebook plots cap-hit and
effective-sample-size fractions because capping introduces bias.

Three-player poker is constant-sum but not a two-player zero-sum game. Low
external regret gives a weaker multiplayer/time-average guarantee, and the
product of three independently averaged policies is not guaranteed to converge
to a Nash equilibrium. Treat the output as an empirical self-play bot and use
league/frozen-opponent evaluation before calling it strong.

## Checkpoints and deployment artifacts

`trainer.save(path, include_buffers=True)` stores all six networks, optimizer
states, both reservoirs for every player, iteration metrics, configuration, and
RNG state, including the variable-stack configuration and sampled continuation
root reservoir. Buffers are required for a faithful CFR resume and can make
files large. `include_buffers=False` creates a smaller inference-only snapshot.

The production campaign keeps the newest three versioned full checkpoints by
default and an atomic `latest.json` manifest. Reservoir fields are serialized
as compact contiguous tensors instead of millions of tiny tensor records.
`Ctrl+C` triggers an emergency checkpoint. If interruption happens between
traversal and fitting, the next fresh kernel detects the incomplete fit and
refits all six networks from the saved cumulative reservoirs before collecting
another traversal. Historical
league snapshots and `champion_policy.pt` contain only the three deployable
average-policy networks. The champion is selected by a confidence-adjusted
composite validation score; it still requires an untouched human-match test
before it can credibly be described as human-beating. Policy-only snapshots now
record their encoder/tournament metadata so the GUI can select legacy or
expanded encoding by the stored input width and flags.

The engine exposes nine discrete bet actions. Real no-limit human deployment
must either enforce that same action set or add a separately tested arbitrary
bet translation/re-solving layer.

Only load checkpoints you trust: PyTorch checkpoints can contain serialized
Python data.
