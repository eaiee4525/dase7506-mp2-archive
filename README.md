# DASE7506 MP2 — Selecting Reliable LLM Reasoning

This repository contains the code, checkpoint, experimental records and report
for my DASE7506 Mini-Project 2 submission. The task is to select one correct
solution from four candidate solutions to each GSM8K question, using the
provided candidate texts and embeddings.

The course instructions are preserved in [`GUIDE.md`](GUIDE.md). Installation
and running instructions are also available in [`code/README.md`](code/README.md).

## Final result

The final predictor uses the fixed Phase14 probability fusion. Its models,
calibrators and fusion weights were frozen before the single official Test
evaluation.

| Evaluation | Correct / Total | Accuracy |
|---|---:|---:|
| Validation, overall | 746 / 798 | 93.4837% |
| Validation, disagreement questions | 303 / 355 | 85.3521% |
| Official Test | **387 / 467** | **82.86937901498929%** |

The submitted value is the official scorer's `accuracy` field,
**82.86937901498929**. Higher is better. All Test questions contain at least
two distinct valid parsed answers; Validation also contains easier questions,
so its overall accuracy is not directly comparable with Test accuracy.

The final report is available at
[`report/DASE7506_MP2_report.pdf`](report/DASE7506_MP2_report.pdf).

## Frozen predictor

The final system combines three answer-group probability distributions:

- **BF (25%)**: a linear embedding scorer trained with valid-candidate BCE on
  disagreement questions, combined with a shallow Format/Final feature scorer
  and an answer-group calibrator;
- **CF (25%)**: a linear embedding scorer trained with group cross-entropy on
  all training questions, combined with the same Format/Final scorer and a
  separate answer-group calibrator;
- **Direct-LME (50%)**: a directly optimized linear group-selection model using
  embeddings, Format/Final features and learned answer-frequency support.

Each branch uses Log-Mean-Exp pooling within valid answer groups. The final
predictor averages complete group probabilities with weights 0.25/0.25/0.50,
selects the highest-probability group, and returns its earliest candidate index.
Equivalent numerical answers are grouped using the unchanged official parser.

The checkpoint contains all weights, preprocessing statistics, calibrators and
configuration needed for inference. It does not require retraining, training
references or intermediate checkpoints.

## Environment and installation

The recorded runs used Python 3.12.14 and these pinned dependencies:

- NumPy 2.5.3
- SciPy 1.18.1
- threadpoolctl 3.7.0

From the repository root:

```bash
cd code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Obtain the official data from
https://github.com/xudongwu-0/DASE7506-MP2 and set its `code/data` location:

```bash
export MP2_DATA="/path/to/DASE7506-MP2/code/data"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_THREAD_LIMIT=1
mkdir -p local_runs
```

Official data are not duplicated in this repository. Inference runs on CPU and
does not require external pretrained weights, an API key or network access.

## Direct evaluation without retraining

All inference assets are included in `code/assets/`. Run from `code/`:

```bash
python -B official/predict.py --method submission.py --checkpoint assets \
  --split test --data-dir "$MP2_DATA" --output local_runs/test_predictions.json
python -B official/score_predictions.py --split test --data-dir "$MP2_DATA" \
  --predictions local_runs/test_predictions.json --output local_runs/test_score.json
```

The expected official result is:

```text
correct = 387
n = 467
accuracy = 82.86937901498929
protocol = dase7506-mp2-open-hard-verifier-v5
```

Use fresh output filenames on repeated runs. Timing fields vary across machines.
Prediction and scoring are separate processes; only the scorer reads references.
For Validation, replace `--split test` with `--split validation` and use new
output filenames. Unlabeled Audit inputs are supported by the prediction runner
with `--split audit --data-dir /path/to/audit`.

## Checkpoint

The complete checkpoint is available in [`code/assets/`](code/assets/) or as a
[downloadable ZIP](https://github.com/eaiee4525/dase7506-mp2/releases/download/v1.0.0/phase14_checkpoint_v1.zip).
Extract the ZIP inside `code/` to obtain the `assets/` directory.

The ZIP has SHA-256:

```text
042f47482444c01121f112bccf942bf7aa1370cbb4c01096ad6d917b6eeecb02
```

Individual code and asset hashes are recorded in [`SHA256SUMS`](SHA256SUMS).
Verify them from the repository root:

```bash
shasum -a 256 -c SHA256SUMS
```

## Optional training reproduction

Retraining is not required to reproduce the submitted score. To reproduce the
fixed Full-Train training procedure, run from `code/`:

```bash
python -B train.py --data-dir "$MP2_DATA" --output local_runs/retrain_001
```

The output directory must not already exist. The resulting inference assets
are written to `local_runs/retrain_001/assets`.

The B, C and Format/Final scorers use seed 20261029, 900 Adam updates, batch
size 128, learning rate 0.001 and L2 regularization 0.0001. Three-fold internal
OOF predictions, with calibration seed 20261130, train the BF and CF
calibrators. Each held-out score is generated outside its base model's fit
region. Final base models are then refitted on their permitted full-Train
populations. Direct-LME uses its fixed MEAN-initialization then LME optimization.
Exact configuration and seeds are in
[`code/frozen_training.json`](code/frozen_training.json).

Floating-point and backend differences may prevent bitwise-identical retraining
across platforms. Use the supplied frozen checkpoint for score reproduction.

## Resource measurements

The recorded evaluation used a local macOS ARM64 machine with 10 logical CPUs,
Python 3.12.14, one configured numerical thread and sequential processes.

| Measurement | Local result | Official limit |
|---|---:|---:|
| Fresh complete Test prediction | 0.564 s | — |
| Fresh independent scoring | 0.076 s | — |
| Complete prediction plus scoring | 0.640 s | 30 s on reference CPU |
| Uncompressed inference assets | 61,839 bytes | 16 MiB |
| Peak child-process RSS | 131.6 MiB | — |

Process-level timing includes startup, imports, input and asset loading,
preprocessing and all prediction branches. The reference Xeon Platinum 8457C
has not been measured; these local results do not certify reference-machine
eligibility. Detailed measurements and training cost are in
[`results/resource_measurements.json`](results/resource_measurements.json).

The supplied data, numeric parser and scorer were not modified.

## Repository contents

- `code/submission.py`: official `predict(bank, arrays, checkpoint)` interface;
- `code/mp2_model/`: candidate scoring, features, answer-group decisions and training;
- `code/train.py`: fixed Full-Train training and OOF calibration;
- `code/frozen_training.json`: training configuration and seeds;
- `code/assets/`: complete frozen inference checkpoint and configuration;
- `code/official/`: unchanged official parser, prediction runner, scorer and data notices;
- `report/`: final report, including AI assistance disclosure;
- `results/evaluation/`: official evaluation outputs;
- `results/experiments/`: selected experimental evidence supporting the report;
- `results/resource_measurements.json`: local evaluation resources and training cost.

Development runs, intermediate checkpoints, datasets, virtual environments and
cache files are excluded from this repository.
