# DASE7506 MP2 — Selecting Reliable Mathematical Answers

This repository contains the submitted selector, the supplied data,
reproduction and training code, a research report, and aggregate experiment
records. The method selects one of four supplied mathematical solutions.

## Submitted model

| Item | File or value |
|---|---|
| Official full-test accuracy | **83.72591006423983% (391/467)** |
| Method | [`code/submission_peer_75.py`](code/submission_peer_75.py) |
| Checkpoint | [`code/runs/peer-test/peer-bottom.npz`](code/runs/peer-test/peer-bottom.npz) |
| Official scoring record | [`code/runs/peer-test/peer-bottom-peer75-test-score.json`](code/runs/peer-test/peer-bottom-peer75-test-score.json) |
| Report | [`report/DASE7506_MP2_Report_peer75.pdf`](report/DASE7506_MP2_Report_peer75.pdf) |
| Detailed instructions | [`code/README.md`](code/README.md) |

The method combines 25% of the standardized original selector with 75% of a
standardized peer-inspired ensemble. Earlier public-test aggregate scores were
known during development; the report explains the limitations of this
comparison. Correctness is evaluated by the official final-answer scorer, not
by verification of every reasoning step.

## Reproduce the result

With Python 3.12, from the repository root:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-experiments.txt
python check_data.py
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
python predict.py --method submission_peer_75.py \
  --checkpoint runs/peer-test/peer-bottom.npz \
  --split test --output runs/peer75-reproduction/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/peer75-reproduction/test-predictions.json \
  --output runs/peer75-reproduction/test-score.json
```

The checkpoint is self-contained and scoring does not require retraining.
LightGBM may also require an OpenMP runtime on macOS. Training, checks, seeds,
local resource use and the historical freeze are described in
[`code/README.md`](code/README.md).

## Contents

- `code/`: the official evaluation tools, final inference and its Python
  dependencies, training scripts, supplied data, matching checkpoints and
  aggregate experiment records.
- `report/`: the six-page report and editable source.
- `guide/`: the original course requirements.
- `SHA256SUMS`: hashes for included Python files, dependency manifests, the
  data manifest and the final checkpoint.

Some dependency modules have names associated with earlier experiments. They
remain because the submitted inference or training code imports shared
functions from them; unrelated experiment outputs and per-question
predictions are excluded.

## Attribution and AI disclosure

The course supplied the data and fixed candidate representations. See
[`code/data/LICENSE.txt`](code/data/LICENSE.txt) and
[`code/data/NOTICE.txt`](code/data/NOTICE.txt). The student proposed the outer
25/75 blend and directed the error analysis. GitHub Copilot was used
substantively for development, experimentation and report drafting. Another
AI-generated summary of online material informed the ensemble design; its
original source list was not retained. The implementation is an approximation,
not an exact reproduction of an identified external model.
