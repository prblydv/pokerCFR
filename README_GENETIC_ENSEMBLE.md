# Weighted HU ensemble genetic optimizer

The optimizer searches non-negative model weights that sum to 100%, with two to
five active policies by default. It does not change the poker engine, legal
actions, policy networks, or top-three stochastic sampling semantics.

## Start or resume

From PowerShell:

```powershell
python -u .\optimize_heads_up_ensemble_genetic.py
```

Or double-click `run_genetic_ensemble_optimizer.bat`.

Re-running the identical command resumes the current generation, validation,
or final holdout stage. A changed configuration must use a new `--output`
directory so results from different searches cannot be mixed.

## Default quality campaign

- Models: 725, 950, 1025, 200, 275, 300, 400
- Maximum active models: 5
- Population: 16
- Generations: unbounded by default; stop safely with Ctrl+C
- Screening: 20,000 reciprocal hands per candidate per opponent
- Validation: best 5 over 200,000 hands per opponent
- Final holdout: best 2 over 1,000,000 hands per opponent
- Fitness opponents: old-three top-3 ensemble and TAG
- Report-only opponent: random, excluded from fitness and penalties
- Random immigrants: 2 every generation to preserve global exploration

All candidates within a generation see common deal seeds. Validation and final
holdout use separate seed ranges.

## Fitness

Fitness remains measured in BB/100:

```text
average league EV
- uncertainty penalty
- penalty for negative non-all-in EV
- penalty for excessive concentration of positive EV in all-in hands
- penalty for the worst losing league matchup
```

The terminal and JSON results show raw EV, all-in EV contribution, non-all-in
EV contribution, standard error, worst-opponent EV, and every individual
penalty. Fitness therefore never hides its construction.

Defaults can be changed with:

```powershell
python -u .\optimize_heads_up_ensemble_genetic.py `
  --non-all-in-loss-penalty 0.50 `
  --max-all-in-positive-share 0.70 `
  --all-in-concentration-penalty 0.50 `
  --worst-opponent-loss-penalty 0.20
```

## Outputs

Default output directory:

`artifacts/downloaded_risk_aware/genetic_ensemble_search_global_v2`

- `events.jsonl`: append-only progress log
- `evolution_state.json`: mid-generation resume state
- `generations/generation_NNN.json`: full candidate and match details
- `generations/generation_NNN.csv`: sortable generation ranking
- `graphs/fitness_and_ev_by_generation.png`
- `graphs/population_fitness_scatter.png`
- `graphs/best_weights_by_generation.png`
- `graphs/best_ev_components_by_generation.png`
- `validation_results.json` / `.csv`
- `final_results.json` / `.csv`
- `best_ensemble.json`: final held-out winner and weights
- `best_live_ensemble.json`: best completed candidate in the active generation

The networks are loaded once. The weighted provider encodes each state batch
once and reuses it across all active networks, avoiding repeated encoder work
without changing probabilities.
