# Installation and running instructions

Run the following commands from `code/`, using Python 3.12.

## 1. Install and prepare data

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MP2_DATA="/path/to/DASE7506-MP2/code/data"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_THREAD_LIMIT=1
mkdir -p local_runs
```

Obtain the official data from https://github.com/xudongwu-0/DASE7506-MP2 and set `MP2_DATA` to its `code/data` directory. The checkpoint is included in `assets/`; no training is needed for prediction.

## 2. Predict and score

```sh
python -B official/predict.py --method submission.py --checkpoint assets \
  --split test --data-dir "$MP2_DATA" --output local_runs/test_predictions.json
python -B official/score_predictions.py --split test --data-dir "$MP2_DATA" \
  --predictions local_runs/test_predictions.json --output local_runs/test_score.json
```

Use fresh output filenames on repeated runs. The frozen official result is **387/467**, with `accuracy` **82.86937901498929**. Its score JSON is in `../results/evaluation/test/score.json`.

For Validation, replace `--split test` with `--split validation` and use new output names. For an unlabeled Audit Bank, run prediction with `--split audit --data-dir /path/to/audit`.

## 3. Reproduce training

```sh
python -B train.py --data-dir "$MP2_DATA" --output local_runs/retrain_001
```

The output directory must not already exist. Exact settings and seeds are in `frozen_training.json`. The trained checkpoint is written to `local_runs/retrain_001/assets`.

## 4. Resources

Inference assets total **61,839 bytes uncompressed**. Local macOS ARM64 Test prediction plus independent scoring took **0.640 seconds**, configured for one numerical thread. The reference Xeon Platinum 8457C has not been measured. Measurements and training cost are in `../results/resource_measurements.json`.
