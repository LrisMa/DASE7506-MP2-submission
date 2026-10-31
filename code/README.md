# Reproduction, training and experiment records

Run commands in this document from `code/`. The inference entrypoint is
`submission_peer_75.py`, and its matching checkpoint is
`runs/peer-test/peer-bottom.npz`. The official score is
`runs/peer-test/peer-bottom-peer75-test-score.json`: 391/467.

## Environment and data

Use Python 3.12 and install `requirements-experiments.txt`. The recorded
environment used Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, scikit-learn
1.9.1 and LightGBM 4.7.0. On macOS, LightGBM may also require an OpenMP
runtime. Run `python check_data.py` to check the supplied data and alignment.

Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS` to 4. The project root README gives exact prediction
and scoring commands. No retraining or test-reference access by the method
itself is needed.

For a validation-only check, use a fresh output directory:

```bash
python predict.py --method submission_peer_75.py \
  --checkpoint runs/peer-test/peer-bottom.npz --split validation \
  --output runs/validation-check/predictions.json
python score_predictions.py --split validation \
  --predictions runs/validation-check/predictions.json \
  --output runs/validation-check/score.json
python -m unittest -v test_peer_75 test_fixed_ratio
```

The previously recorded validation result for this model is 739/798.

## Training and diagnostics

`export_peer.py` fits the ensemble on the training split with seed 7507 and
uses the saved training-derived `runs/numeric/selected.npz` for the old
branch. It creates `runs/peer-test/` and refuses to overwrite an existing
directory. To reconstruct training, use a disposable copy and first move its
existing `runs/peer-test/` directory to a different name; then run:

```bash
python export_peer.py
```

The original old-branch and stacker training routines are included as
`train_answer.py --numeric` and `train_meta.py`. Their existing checkpoints
are included under `runs/numeric/` and `runs/meta/`. They refuse to overwrite
existing output files; use a fresh `--output-dir` for experiments. Refer to
the scripts' `--help` output for options.

Matched three-seed training diagnostics and the numeric-backoff comparison
can be reproduced with fresh output paths:

```bash
python train_peer_diagnostics.py --output runs/normalization/peer-75-repeat.json \
  --seeds 6813 7253 7507
python train_peer_diagnostics.py --numeric-backoff \
  --output runs/normalization/numeric-backoff-repeat.json --seeds 6813 7253 7507
```

The `runs/normalization/` records contain the original aggregated diagnostics.
`runs/fixed-weight-comparison/` contains the fixed-ratio comparison and
aggregate scores; per-question predictions are not included.

To reproduce the seed-6813 training error counts in the report, first create
held-out training component scores, then analyze them:

```bash
python cache_peer_components.py --seeds 6813
python analyze_train_errors.py
```

These are training-only intermediate scores, not inference assets or cached
test predictions. Equation diagnostics can be reproduced with
`python train_equation_diagnostics.py --output runs/normalization/equation-repeat.json`.

## Integrity and resources

From the repository root, `shasum -a 256 -c SHA256SUMS` verifies the included
Python sources, requirements, data manifest and final checkpoint. The final
checkpoint SHA-256 is
`a174829d7952d347f21e1538ae612dfbb2ec30a9350e467e3cf2d81fe1d54529`.

The original pre-test freeze and local resource record are in `runs/peer-test/`.
The historical freeze lists hashes for the larger development workspace,
including files not needed by this trimmed package; `SHA256SUMS` covers the
files supplied here. On an Apple M1, the recorded complete test prediction
plus independent scoring took 8.96 seconds, and the checkpoint contained
10,235,918 uncompressed array bytes. Timing on the required Xeon reference
CPU remains unverified.

The data and official parser are unchanged. During development, earlier
aggregate public-test results were known and validation was reused. The
report describes the comparison and its limitations.
