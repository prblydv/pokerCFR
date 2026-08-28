# Fresh heads-up CFR training on Vast.ai

This directory is the clean migration package for a new heads-up campaign. It
contains no three-player code, old checkpoints, generated artifacts, or Windows
binaries.

If `heads_up_vast_migration.zip` was uploaded to `/workspace`, unpack it into a
fresh source directory without deleting the previous campaign:

```bash
cd /workspace
if [ -d heads_up_training ]; then
  mv heads_up_training "heads_up_training_pre_v3_$(date +%Y%m%d_%H%M%S)"
fi
mkdir heads_up_training
unzip heads_up_vast_migration.zip -d heads_up_training
cd heads_up_training
```

Rent one GPU with at least 16 GB VRAM, 12 effective CPU cores, 64 GB system
RAM, and a clean 80 GB fast NVMe instance volume. Copy this directory to
persistent Vast.ai storage, then run:

```bash
cd /workspace/heads_up_training
bash setup_vast.sh
```

Use an Ubuntu PyTorch image with SSH/Jupyter support and sufficient persistent
disk. The setup script installs the Python requirements, compiles the heads-up
C++ engine for the instance's Python ABI, runs the full correctness suite,
verifies the hidden-384 dual-head architecture (1,842,512 parameters per
advantage network, 2,353,022 per policy network, 8,391,068 total), and runs a real CUDA matrix
multiplication.

Start Jupyter:

```bash
tmux new -s heads-up-jupyter
bash start_jupyter.sh 2>&1 | tee jupyter.log
```

Open the 36-cell `heads_up_training.ipynb` and run all cells. Its production
campaign has the same logging, evaluation, league/champion tracking, charts,
versioned checkpoints, configuration history, resume manifest, and recovery
behavior as the three-player notebook, adapted for two seats and HU utility.
By default it starts at
iteration 0, targets iteration 10,000, collects 1,024 traversals per player
with 16 traversal workers by default on the 16-core Ryzen 9 7950X,
evaluates random/calling-station/TAG and the frozen iteration-1025 policy every
25 iterations. At the same cadence it evaluates the 1,326-combination
opponent-range output on a fixed no-leakage holdout and saves NLL,
information-gain, top-10/top-50, rank, entropy, street, position, and opponent
graphs,
and saves a full resumable checkpoint every 25 iterations. Each of the two
advantage and two policy reservoirs retains up to eight million uniformly
sampled entries in packed system-RAM tensors. Policy samples also retain the
revealed opponent-combination label, but the encoded network input never
contains those hidden cards.

After any advantage or policy buffer reaches eight million entries, the next
new sample removes the oldest 18% (1,440,000 entries). Every subsequent new
sample is retained until the buffer reaches eight million again, then the same
FIFO turnover repeats. The checkpoint stores the logical ring position and
cumulative turnover counters.

Worker samples are copied into the rings in exact-order tensor batches instead
of one Python row at a time. The two independent seat networks are fitted
concurrently on separate CUDA streams for the advantage phase and again for the
policy/range phase. This retains the configured 8,192 batch size, 977 Adam
updates, losses, clipping, samples, and FIFO turnover semantics.

The fresh campaign writes to `artifacts/heads_up_v4_hidden384/` and reads
`reference_policies/policy_00001025.pt` only as an evaluation opponent.
It therefore cannot accidentally resume or train from the hidden-128 campaign.

A mature full checkpoint with all four eight-million-entry packed reservoirs is
approximately 65-70 GiB. Atomic replacement temporarily needs both the previous
checkpoint and the new checkpoint, so keep at least 145 GiB free while saving.
The notebook retains only one full checkpoint; policy snapshots and evaluation
CSVs are much smaller. Do not put unrelated datasets or old campaigns on the
training volume.

For unattended execution:

```bash
tmux new -s heads-up-training
bash run_training_headless.sh
```

Override notebook settings with environment variables, for example:

```bash
POKER_TARGET_ITERATION=20000 POKER_TRAVERSALS=16 bash run_training_headless.sh
```

Monitor with `bash training_status.sh`. Before destroying the instance,
download the complete `artifacts/heads_up_v4_hidden384/` directory. It contains
versioned full checkpoints, policy snapshots, raw evaluation hand CSVs,
`metrics.jsonl`, `latest.json`, the champion, and run-configuration history.
