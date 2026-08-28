# Risk-aware hidden-512 HU campaign

This package starts a fresh, isolated campaign in
`artifacts/heads_up_risk_aware_hidden512_v1`. It does not contain or resume any
older checkpoint or reservoir.

Training contract:

- restored 1,038-feature HU information state and ten legal policy actions;
- hidden-512 `hu_deep_cfr_compact_v4` advantage and policy networks;
- no opponent-range head, range reservoir, range collection, or range loss;
- all actions, including all-in, remain legal and are traversed;
- risk-aware regret shaping scrutinizes all-ins above a 2x
  payment/pot-after-call ratio and reduces only marginal positive shove regret;
- Smooth-L1 advantage loss;
- one shuffled pass over every current advantage and policy reservoir per
  iteration, with a dynamic final batch and no row repetition;
- advantage networks/Adam continue through iterations 1-24, reset before the
  fit at iteration 25, then reset from scratch again at every later iteration;
- evaluation every 25 iterations against random, calling-station, TAG,
  policy-1025, and the 725/950/1025 top-three ensemble;
- `all_in_spr_trend.png` reports median and p90 SPR only at decisions where the
  candidate actually chose all-in.

On a new Vast instance:

1. Upload and extract the archive.
2. Run `bash setup_vast_risk_aware.sh`.
3. Start Jupyter using `/venv/main/bin/python -m jupyter lab --ip=0.0.0.0`.
4. Open `heads_up_risk_aware_training.ipynb` and Run All.

Before destroying the instance, copy the complete fresh artifact directory to
durable storage. Full checkpoints contain resumable reservoirs; policy
snapshots do not.
