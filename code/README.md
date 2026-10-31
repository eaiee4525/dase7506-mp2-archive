# DASE7506 MPJ2 — fixed Phase14 final evaluation package

**Official Test: 387/467, accuracy 82.86937901498929%.** This is the exact official scorer `accuracy` field in `../results/evaluation/test/score.json`. The selected model is the unchanged Full-Train Phase14 probability mixture **BF25% + CF25% + Direct-LME50%**. Evaluated submission commit: `6c691a63bf9a6b3c8b1f45cd68e6d8944ebae7f7`. The algorithm hash and individual evaluated source hashes remain in `VERSION.json`; every inference/training source and asset in this package is identical to that evaluated version.

Existing Official Validation:746/798 overall(93.4837%),303/355 disagreement(85.3521%). Historical Train fivefold disagreement OOF on three partitions:1736/1733/1737 out of2114(82.1192/81.9773/82.1665%). Test contains467 disagreement questions. These are different banks and evaluation settings; differences are descriptive, not paired improvements or future audit guarantees. Historical OOF has repeatedly informed research selection. No Test-driven tuning, other-candidate Test evaluation or new training was performed.

## Contents

`submission.py` exposes `predict(bank, arrays, checkpoint)`; `assets/` includes all weights, learned preprocessing, calibrators, frozen config and manifest. `mp2_model/`, `train.py`, `frozen_training.json` and `docs/METHOD.md` provide the complete fixed training/calibration implementation. `official/` contains unchanged course parser/predict/scorer. `../results/evaluation/test/` contains the original official score JSON and resource/version evidence; the frozen prediction hash is retained in provenance (raw prediction artifacts are not required by inference); these are evaluation artifacts and are never loaded by predict. `../results/evaluation/validation_*` retains previous official evidence. `../results/resource/` retains aggregate reproduction/resource evidence. Root `SHA256SUMS` covers this release. The final nine-page Report is `../report/DASE7506_MP2_report.pdf`; selected Report evidence and source hashes are indexed in `../results/experiments/INDEX.json`. No bank inputs, references, answer table or external pretrained weights are distributed.

## Install and data

Use Python3.12 (tested3.12.14). From the `code/` directory:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
(cd .. && shasum -a 256 -c SHA256SUMS)
export MP2_DATA="/path/to/DASE7506-MP2/code/data"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_THREAD_LIMIT=1
mkdir local_runs
```

Obtain official data separately from https://github.com/xudongwu-0/DASE7506-MP2. The data path above is a caller-supplied placeholder, not a machine-specific dependency. NumPy2.5.3, SciPy1.18.1 and threadpoolctl3.7.0 are pinned in `requirements.txt`. Evaluation is offline after installation. New isolated Python processes with installed pinned dependencies were tested; a fresh pip download/install and cross-platform BLAS bitwise reproduction have not been tested.

## Reproduce the frozen official Test score

The authorized final run has already happened once. The following commands are for independent reproduction, not another experiment or automatic rerun. Choose fresh output filenames; the official runner can overwrite output.

```sh
python -B official/predict.py --method submission.py --checkpoint assets \
  --split test --data-dir "$MP2_DATA" --output local_runs/test_predictions.json
python -B official/score_predictions.py --split test --data-dir "$MP2_DATA" \
  --predictions local_runs/test_predictions.json --output local_runs/test_score.json
```

Prediction does not read Reference. Only the separate scorer reads Test Reference after predictions are fixed. Compare protocol/split/IDs/selected and `correct`, `n`, `accuracy` for reproduction; timing fields and whole-JSON hashes can vary across reruns. The original evaluation prediction was hashed before scoring and remained unchanged. Original frozen artifact hashes are in `../results/evaluation/test/provenance.json`; the full exact private commands remain in the project evaluation logs. No Test errors were inspected to modify this model.

The same command with `--split validation` and fresh names reproduces the existing Validation evaluation. For new unlabeled Audit inputs, use `--split audit --data-dir /path/to/authorized/audit` with predict only. No retraining or Reference is needed. The API supports arbitrary input counts/order/IDs, float32 `[N,4,1024]` embeddings and native Python int0–3 outputs; no ID lookup or cached prediction is consulted. `checkpoint=None` uses local `assets/`.

## Training and offline engineering checks

```sh
python -B train.py --data-dir "$MP2_DATA" --output local_runs/retrain_001
python -B verify_interface.py --data-dir "$MP2_DATA" \
  --checkpoint local_runs/retrain_001/assets --output local_runs/retrained_interface.json
python -B check_resources.py --data-dir "$MP2_DATA" --output local_runs/check_001
```

These commands are optional reproduction, not required to use the delivered checkpoint. No training ran during official Test. The fixed training entry fits3 internal Question folds(seed20261130),9 held-out B/C/F models, two calibrators using only legal OOF scores,3 full-fit base models and the fixed Direct MEAN→LME sequence. B uses disagreement population; C/F use all Train. Scalers fit the allowed fit region, invalid/multiple-positive labels are preserved. Seed20261029 and exact settings remain frozen. No in-sample final-model scores fit their calibrator. Prior same-environment training reproduced all exported arrays bitwise; NPZ container hashes can vary with compression metadata. The integrity-only manifest is not needed by trained-model inference.

## Resources and limits

Inference assets: **61,839 bytes uncompressed** including NPY headers/config/manifest (0.05897MiB; disk59,360bytes), below16MiB. Authorized complete467-question Test: fresh prediction0.564251s plus separate official scoring0.075884s = **0.640135s**, including process startup/imports/input+asset loading/features/all experts, access guard and log writes. Peak child RSS137,969,664bytes. Local host: macOS27 arm64,10 logical CPUs; Python3.12.14.

All numeric-thread environment settings are1; inference uses `threadpool_limits(1)` and evaluation processes are sequential. OS threads were not sampled in this run; earlier project OS sampling observed at most1 target thread, with sampling limitations. Local threadpoolctl reports no enumerable runtime on this Mac. The reference Xeon Platinum8457C was **not measured**; local timings do not certify its30-second/four-total-thread qualification.

## Report, sources and submission

Final Report: [DASE7506_MP2_report.pdf](../report/DASE7506_MP2_report.pdf),9 pages. Model code/configuration and all inference assets match the evaluated frozen version. Complete file SHA256 hashes are in the repository-root `SHA256SUMS`; `PUBLICATION.json` binds this version to the evaluated code, checkpoint and Report.

AI/reuse disclosure and source/license notices: `docs/AI_REUSE.md`, `docs/THIRD_PARTY_NOTICES.md`, `docs/LICENSE.md`, `official/DATA_LICENSE.txt`, `official/DATA_NOTICE.txt`. No official data, reference answers, sample-level labels, private development paths or historical checkpoints are distributed. The Report's author/student identification is intentionally part of the author-approved final document.

Final course submission remains the author's action: use the versioned Code/Checkpoint links, the official scorer accuracy82.86937901498929, and create the submission via the course site. This publication does not create a submission Issue or fill the leaderboard. Reference Xeon qualification and fresh target-environment installation remain unverified; local timing is not reference-machine certification.
