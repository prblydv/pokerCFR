# Heads-up no-limit Hold'em migration

Heads-up is the new development target. The existing three-player engine,
training code, checkpoints, and GUI remain separate and unchanged; HU files,
schemas, native binaries, and artifacts use `heads_up_*` names.

## Architecture

- `heads_up_engine.py`: readable integer-chip correctness reference.
- `engine C HU/heads_up_native_engine.cpp`: isolated C++20 implementation.
- `heads_up_native.py`: schema-checked native compatibility facade.
- `heads_up_models.py`: hero-visible information-state encoder and ten-output
  advantage/policy networks.
- `heads_up_cfr.py`: two-player zero-sum external-sampling Deep CFR.
- `train_heads_up.py`: resumable Python/native training CLI.
- `heads_up_reporting.py`: six-panel training dashboard.
- `heads_up_training.ipynb`: fresh/resumable CUDA notebook.
- `heads_up_search.py`: exact observed-action and ten-slot policy/search bridge.
- `play_heads_up_gui.py`: human-vs-policy GUI with an optional manual two-seat mode.

## Exact room actions versus finite policy actions

The engine has two intentionally separate interfaces:

```python
state = env.step_exact(state, "raise_to", raise_to=237)
state = env.step(state, policy_slot)
```

`step_exact` accepts arbitrary legal integer chip targets and never rounds or
clamps them. The exact action, target, payment, pot, full-raise flag, and all-in
flag are retained in semantic hand history.

The trained policy always emits ten values:

| Slot | Policy action |
|---:|---|
| 0 | fold |
| 1 | check |
| 2 | call |
| 3 | minimum raise |
| 4 | one-third pot |
| 5 | half pot |
| 6 | three-quarter pot |
| 7 | pot |
| 8 | one-and-a-half pot |
| 9 | all-in |

For a pot-fraction raise, let `c` be the actor's current street contribution,
`call` the exact call payment, and `P` the pot before acting:

```text
called_target = c + call
pot_after_call = P + call
raise_to = called_target + round_half_up(fraction * pot_after_call)
```

Every target is computed by the engine. A named size is present only when its
exact target is legal. It is never silently clamped to the stack. Slot 9 is the
only abstract slot that may make a sub-minimum all-in raise. Effect-identical
slots are deduplicated, with the lower slot becoming canonical. The engine's
legal mask, descriptors, and transition all consume the same generated action
set, preventing sizing drift.

The finite game therefore stays mathematically well-defined for CFR, while an
observed off-tree room raise stays exact in the state. After such a raise, the
bot still chooses among the ten legal, state-dependent responses.

## Encoder and checkpoint contract

The encoder:

- includes only the acting player's private cards;
- represents both seats relative to the acting player;
- records exact semantic bet/raise history rather than bucket labels;
- normalizes chip quantities by the big blind and adds dimensionless pot/SPR
  ratios;
- includes the exact target, payment, resulting pot/stack, all-in, aggression,
  full-raise, and reopening effect for every legal policy slot;
- embeds the exact ten-bit legal mask;
- rejects a non-acting hero or missing action descriptors.

The default 32-event history produces 1,038 float32 inputs. If a hand exceeds
that capacity, the encoder explicitly retains the most recent 32 events. This
is a bounded-history abstraction rather than perfect recall.
Checkpoints persist the engine, action, encoder, action-order, width, and
history-capacity metadata. The native module has an additional ABI handshake,
so an old `.pyd` fails at import instead of silently changing game semantics.

The production network is `hu_deep_cfr_compact_v4`, derived from the last
three-player `deep_cfr_branch_v3` design. It preserves separate card,
poker-relation, public-state, exact-action-descriptor, and ordered-history
branches; one-layer Transformer encoders for history/actions; a one-layer GRU
history memory; a fused residual trunk; and independent
preflop/flop/turn/river action heads. The production configuration uses hidden
width 128 with two two-layer residual blocks, 32-wide card embeddings, and
32-wide history/action tokens. Each advantage or policy network has 294,736
parameters, just under three times the 98,948-parameter network in the original
Deep CFR paper and about 9.7 times smaller than the prior HU V3 network. All
made-hand, draw, exact-action, GRU, and street-head information is retained.

## Play against a policy

```bat
run_heads_up_gui.bat
```

The launcher defaults to the downloaded Vast iteration-1025 snapshot with
three-player root search, a policy-likelihood inferred human range, 0.65
likelihood temperature, 25% uniform contamination, a 1,350 ms internal
budget (approximately 1.5 seconds of observed decision time), 1,024-iteration
batches, a 150,000-rollout cap, and a 10% minimum final-strategy sampling
probability. Results are
appended after every completed hand to
`artifacts/heads_up_gui_results/hands.jsonl`. The GUI shows lifetime human and
policy BB/hand for the active policy and has a **Policy history** window for
comparing all policy fingerprints and play modes.

New rows use the append-only `heads_up_gui_hand_v2_complete` record schema.
Each completed-hand row contains both players' hole cards, public board,
burned and remaining cards, the exact reconstructable 52-card deck order,
structured action events with precise `raise_to`, pot, stack, contribution,
board-at-action, and all-in fields, terminal payouts/payoffs, every root-search
candidate and confidence interval, the active controller/search configuration,
and inferred-range diagnostics.  Range updates also record the posterior
probability and rank assigned to the human's actual combination and hand
class; these audit-only fields are not shown to or used by the bot during the
hand.  Version-1 historical rows remain readable.

The complete deck and exact action sequence allow later replay and
all-in-equity/runout analysis.  Recording the data does not itself remove
variance; a later analyzer can compare observed payoff with equity-adjusted
payoff at decision-lock or all-in points.

To select another snapshot or seat:

```bat
python play_heads_up_gui.py --policy path\to\policy.pt --human-seat 1
```

For the original manual two-seat engine mode, omit `--policy`:

```bat
python play_heads_up_gui.py --stack 200 --sb 1 --bb 2
```

Manual mode shows both hands. Human controls expose every legal finite slot
with its exact target and accept an arbitrary integer `raise_to`.

## Controlled policy scenario suite

`evaluate_heads_up_policy_scenarios.py` legally replays a permanent catalogue
of 69 important preflop, flop, turn, and river decisions.  It evaluates every
case through both physical seat networks, records the complete ten-action
distribution, checks illegal probability mass and seat consistency, applies a
small set of broad hand-strength ordering checks, and measures strategy drift
between consecutive checkpoints.

```bat
python evaluate_heads_up_policy_scenarios.py
```

The default comparison is iteration 350 versus 725 versus 950 and writes CSV
tables plus `summary.json` under
`artifacts\heads_up_v4_paper3x\evaluations\scenario_suite_350_725_950`.
These are controlled strategy fingerprints and behavioral sanity checks.  They
do not by themselves measure exploitability or prove that a poker action is
optimal; attach an independent action-EV oracle before using a scenario as a
strict poker-quality gate.

## Approximate exploitability

`evaluate_heads_up_exploitability.py` freezes each target policy and trains a
separate one-sided external-sampling Deep CFR response for each physical seat.
It selects the response checkpoint and safe blueprint/response mixture on
validation deals, then evaluates once on an untouched reciprocal deal seed.

```bat
python evaluate_heads_up_exploitability.py
```

This is a learned-best-response lower bound, not exact HUNL exploitability.
Failure to find a winning response is inconclusive.  A positive result with a
positive confidence-interval lower endpoint demonstrates an exploit, while a
zero result only means that this response learner and budget did not find one.
The response uses the production information-state encoder and cannot observe
the frozen opponent's private cards.

## Build the native engine

```bat
"engine C HU\build.bat"
```

The checked-in binary was built for Windows CPython 3.13. Rebuild it for any
other Python ABI.

## Train

Quick smoke run:

```bat
python train_heads_up.py --engine auto --iterations 1 --traversals 1 --adv-steps 1 --policy-steps 1 --batch-size 8 --eval-every 1 --eval-games 2 --save-every 1
```

Normal entry point:

```bat
python train_heads_up.py --engine auto --device cuda --iterations 100
```

`auto` prefers the native C++ engine and falls back to the Python reference.
Full checkpoints include both reservoirs and can resume faithfully. A light
checkpoint omits reservoirs and is inference-only.

The compact production defaults use 1,024 traversals per player, 16 CPU
traversal workers on the 16-core Ryzen 9 7950X, and 977 advantage and policy update steps
(enough to cover each eight-million-entry reservoir once), batch size
8,192, hidden width 384, two two-layer residual blocks, learning rate
1e-3, persistent advantage networks through iteration 24 followed by fresh
cumulative refits from iteration 25 onward, evaluation every 25 iterations
against random/calling-station/TAG plus the frozen iteration-1025 policy, and
eight million samples in each physical advantage/policy reservoir. The
iteration-1025 policy is an evaluation-only opponent: it does not initialize or
populate the new networks or reservoirs.

The 36-cell HU notebook now uses the same production-campaign control flow as
`three_player_training.ipynb`, with HU-specific seats, engine, utility, and
opponents. It keeps the same live iteration logging, fixed-seed evaluation
cadence, raw hand CSV exports, historical-policy league, champion promotion,
nine-panel charts, periodic strategy reports, versioned full checkpoints,
`latest.json` resume manifest, append-only `metrics.jsonl`, run-configuration
history, and emergency/failed checkpoint recovery. The fresh hidden-384
campaign writes to `artifacts/heads_up_v4_hidden384`; the previous hidden-128
campaign remains separate.

The CLI remains a smaller standalone entry point and writes
`heads_up_metrics.json` plus `heads_up_training_dashboard.png`.

For a clean Linux/CUDA migration, copy `vast_heads_up_training/` to the Vast.ai
machine and follow its `README_VAST.md`. It intentionally excludes old
three-player code, artifacts, checkpoints, and Windows binaries.

The trainer alternates both traversers and both button positions, enumerates
all legal actions at traverser nodes, samples the opponent at external-sampling
nodes, trains two advantage networks, and trains two reach-sampled
average-policy networks. Terminal utility is net chips divided by the big
blind. Strict mode defaults to zero fixed exploration. `--exploration` can add
a uniform coverage mixture, but any nonzero value introduces an approximation
floor and is recorded in the checkpoint.

The production hidden-384 campaign fresh-initializes both advantage networks
at iterations 25, 50, 75, and so on. Within each 25-iteration cycle, the
networks and Adam optimizer state continue training instead of restarting.
The cycle length is checkpointed and explicitly overridden to 25 when an older
campaign checkpoint is resumed through the notebook.

The fresh hidden-384 campaign uses a dual-head policy network. Its shared
policy trunk feeds the existing 10-action head and a blocker-masked
1,326-exact-combination opponent-range head. The fitting objective is
`policy_action_loss + 0.01 * range_loss`: action loss directly trains only the
action head, range loss directly trains only the range head, and both update
the shared policy trunk. Advantage/regret networks are unchanged. Action
targets still come only from the CFR policy reservoirs. Range targets now come
from separate, independently dealt single-trajectory hands against self-play,
random, calling-station, tight-aggressive, and frozen-reference opponents.
Each hand follows only the action actually sampled; counterfactual CFR branches
never enter the range reservoirs. Per-hand weights stop long hands from
dominating. The encoded input remains the decision-time information set, so
hidden cards, future actions, and future board cards cannot leak into the input.
Legacy dual-head checkpoints retain their CFR trunks/action heads and
reservoirs, but reset only the old range projections before populating the new
range replay.

All six hidden-384 training buffers use chunked recency turnover rather than
lifetime-uniform reservoir replacement. A buffer appends normally until it is
full. The next incoming sample evicts the oldest 18% as one logical FIFO chunk,
then every newly generated sample is appended until the buffer is full again;
the cycle repeats. Checkpoints preserve the ring position and turnover counters,
so resumed training retains the exact age ordering.

Parallel worker results are merged into those FIFO rings with exact-order
batched tensor copies rather than Python row-by-row insertion. This changes
only the transfer mechanism: sample values, worker/result order, eviction
boundaries, and logical buffer order remain the same. On CUDA, the independent
player-0 and player-1 advantage fits run concurrently on separate streams, then
the two policy/range fits do the same. Batch size, fitting-step count, losses,
gradient clipping, optimizer type, and per-player parameters are unchanged.
Every evaluation also writes `range_reservoir_dashboard.png` and a JSON
snapshot containing street percentages, the complete 13x13 opponent
starting-hand matrix, and made-hand categories overall and by street.

## Off-tree search contract

Use `apply_observed_action` to consume a room action exactly, then
`HeadsUpNetworkPolicy` or `build_decision_context` to obtain a schema-checked
ten-slot response:

```python
from heads_up_search import apply_observed_action, build_decision_context

state = apply_observed_action(env, state, "raise_to", 237)
decision = build_decision_context(env, state)
```

The included `HeadsUpRealTimeResolver` is a bounded Monte Carlo improvement
heuristic. It preserves exact actions and ten-output legality, but samples the
opponent's hidden cards uniformly. It is not a Pluribus-equivalent,
range-aware safe re-solver.

The human-vs-policy GUI additionally enables
`MultiprocessPluribusSearch` by default. Worker count is selected from physical
cores with one core reserved for GUI/holdout coordination (five workers on the
local 12-logical-CPU machine). The coordinator budget is 6 seconds and the
public-tree depth limit is three decisions:

```bat
run_heads_up_gui.bat
```

The search compares the legal blueprint actions plus exact integer root sizes
generated around 25%, 33%, 50%, 60%, 71%, 75%, 100%, 125%, 150%, and 200% of
the pot after calling (with minimum raise and all-in included). Any selected
off-tree size is applied with `step_exact`; it is not relabelled as a neural
action bucket.

At the start of every hand, the opponent range is blocker-uniform. Each public
opponent action updates the current hand's combination weights by its frozen
blueprint likelihood. Exact off-tree sizes use a smooth likelihood over nearby
blueprint raise sizes while the engine transition remains exact. The range is
discarded at the next hand; there is no persistent opponent personality,
mapping, or framing.

Workers alternate CFR traversers. Root counterfactual actions use common random
numbers. At the depth boundary, each player can select among four continuation
strategies: normal blueprint, fold-biased, call-biased, or raise-biased. Search
workers use the compiled C++ state clone, legality, transition, exact-bet,
runout, and showdown path while neural inference remains batched PyTorch.

The policy network supplies leaf continuations and root priors; it never owns a
searched decision and is never blended back into the final root strategy.
Workers solve in two synchronized waves. The second wave inherits the shared
root regret table and uses successive elimination to spend work only on
statistically plausible actions. Agreement and regret matching are hierarchical:
fold, call/check, or raise first, followed by an exact raise-size solve. Street
depth is adaptive: wider and shallower preflop/flop, deeper on turn/river.
Neural leaves are evaluated in batches and native state transitions are also
advanced in batches.

A separate paired holdout identifies statistically dominated actions. A solve
is considered stable only when it has sufficient root/validation samples, at
least 60% action-family agreement, and a two-point family-strategy gap.
Unresolved decisions remain search-owned and are ranked by search and validation
EV; they are not forced into passive blueprint-prior actions.

Postflop actions that commit at least four current pots and half the remaining
stack require positive evidence above a conservative risk floor. Root all-ins
receive an additional blocker-aware C++ evaluation. That evaluator estimates
equity separately for every opponent range hand and applies a rational-calling
floor, so an excessively tight blueprint calling range cannot manufacture a
profitable shove.

When the bot faces a terminal postflop fold/call decision, another C++ guard
prices the call against four range scenarios: the Bayesian posterior, a
tempered posterior, a uniform-contaminated posterior, and a value-heavy stress
range. The call is removed unless its lower bound beats folding under the
worst plausible scenario. This protects against a blueprint likelihood model
that incorrectly assumes a human shove range is bluff-heavy.

After every searched bot action, the exact public-history panel shows the
chosen mixed-strategy probability, CFR diagnostic payoff, independent holdout
payoff/confidence interval, safety-pruning status, public-range size/effective
sample size, CFR iterations, continuation rollouts, native-backend status, and
the exact sample/convergence reason.

The GUI process removes the human hole cards, burn cards, and deck order before
submitting work to search processes. A completed search always owns the action.
If search fails before returning any legal solution, the GUI uses a
deterministic passive emergency action (check, then call, then fold); it does
not query the policy network. To compare against the raw network deliberately,
launch with `--no-search`. Worker count and deadline can be overridden with
`--search-workers` and `--search-budget-seconds`; zero workers means
hardware-aware automatic selection and the deadline is restricted to at most
12 seconds.

GUI search is selectable without changing checkpoints:

```powershell
# Three-player-style root resolver with inferred human range and the current
# 65% policy / 35% search GUI anchor.
python play_heads_up_gui.py --policy artifacts/heads_up_v4_paper3x/snapshots/policy_00001025.pt --policy-device auto --search-mode three-player --root-range-mode inferred --root-blueprint-weight 0.65

The three-player-style HU root resolver batches paired continuation samples,
uses CUDA automatically when available, and advances fully determinized
rollouts through the native C++ engine when the extension is installed. Native
ABI 5 encodes each actor-homogeneous frontier into one contiguous observation
and legal-mask pair, samples supplied paired thresholds on the GPU, and advances
the selected actions through one native `step_batch` call. Run
`benchmark_heads_up_root_search_batching.py` to compare this path with the
legacy scalar CPU/Python loop on the same policy and root state.

# Separate distributionally robust resolver. The original mode remains intact.
python play_heads_up_gui.py --policy artifacts/heads_up_v4_paper3x/snapshots/policy_00001025.pt --search-mode robust --robust-action-noise 0.10 --robust-kl-radius 0.20

# Family-shared depth-limited HU CFR resolver.
python play_heads_up_gui.py --policy artifacts/heads_up_v4_paper3x/snapshots/policy_00001025.pt --search-mode cfr
```

The default `run_heads_up_gui.bat` launch is an isolated plain-policy campaign.
It disables search, samples the complete legal network distribution without a
minimum-probability exclusion or premium-hand override, and records hands plus
persistent HUD statistics under
`artifacts/heads_up_gui_results/plain_policy_sample_campaign/hands.jsonl`.
Supplying explicit arguments to the batch file still permits other GUI modes.

The `three-player` mode retains the resolver's 150,000-rollout cap,
64-action rollout ceiling, paired terminal rollouts, KL-bounded refinement,
and mixed-strategy sampling. The GUI defaults to a 65% policy anchor and 35%
search weight; `--root-blueprint-weight` controls that split.
The bot's first voluntary preflop decision in each hand always comes directly
from the raw policy and bypasses search. A second preflop decision after a
re-raise, and every postflop decision, uses the configured search mode.
As a GUI-only safety rule on that first preflop policy action, `AA`, `KK`,
`QQ`, `JJ`, and all suited or offsuit `AK` combinations cannot fold. Fold
probability is removed and the remaining raw-policy probabilities are
renormalized. The rule does not apply to a second preflop decision; search
controls that response normally.
`--root-range-mode inferred` samples a temperature-0.65 Bayesian
range derived from exact observed-action likelihoods with 25% uniform
contamination; `--root-range-mode uniform` restores the original uniform hidden
card determinizations. Neither mode reads the human hole cards. In inferred
mode, the GUI's right panel displays a live 13x13 range heatmap after every
human action: pairs are diagonal, suited classes are above the diagonal,
offsuit classes are below it, and dark-to-gold color represents posterior
class probability.

The separate `robust` mode does not alter the default `three-player` resolver.
It conditions the public range with a generic action-space tremble rather than
uniform hidden-card contamination, scores paired root payoffs using a
KL-bounded worst-case reweighting across opponent-hand particles after chance
outcomes have been averaged, adds small opponent continuation trembles, and
selects the maximin root action. The KL adversary cannot choose bad deck
runouts. Turn and river decisions that are limited to folding or calling an
all-in enumerate every blocker-compatible opponent hand and remaining public
card exactly, average runouts per hand, and only then reweight hands.
`--robust-action-noise` controls the action-model error mass and
`--robust-kl-radius` controls the ambiguity set; both parameters are isolated
to `--search-mode robust`.

This is a practical Pluribus-style depth-limited CFR approximation, not the
published Pluribus implementation and not a proof of safe continual resolving.
It uses neural blueprint likelihoods, native batched traversal hot paths,
independent holdout validation, and synchronized worker solvers; Pluribus used a
much faster, heavily optimized tabular/abstract subgame stack.

That distinction matters: Deep CFR here solves the fixed ten-action abstract
game. An arbitrary opponent raise is represented correctly but can still be
out-of-distribution for the blueprint. The search preserves the exact state
and uses a sizing-likelihood bridge, but that does not retroactively make the
neural blueprint an exact continuous-action solution.

## Verification

```bat
python -m unittest -v test_heads_up_engine.py test_heads_up_models.py test_heads_up_native_engine.py test_heads_up_search.py test_heads_up_pluribus_search.py test_heads_up_training.py
python -m unittest -v test_three_player_engine.py
```

The HU suite covers dealing and action order, arbitrary raises, strict
minimum/short-raise rules, check-raises, all-ins, automatic runouts, uncalled
returns, odd chips, evaluator behavior, immutability, randomized conservation
and zero-sum invariants, native/Python differential paths, encoder privacy and
schema guards, search contracts, CFR collection, legal masking, and checkpoint
resume.

Current deliberate exclusions are rake, antes, straddles, and multiple
runouts. Add them only as new versioned engine/action schemas.
