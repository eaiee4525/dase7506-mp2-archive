# Frozen model and Full-Train reproduction

Version `phase14-full-train-v1`; group probabilities are mixed in the original arithmetic order `((P_BF + P_CF)/2 + P_DirectLME)/2`. Weights are BF .25, CF .25, Direct-LME .50. Temperature is 1; no new gate or hyperparameter is fitted.

B is a 1024-dimensional linear BCE-valid scorer optimized on 2,114 disagreement Train questions. C is a linear answer-group CE scorer trained on all 4,755 questions. F is the fixed 20-dimensional format/final representation with a 32-ReLU head (historical 116-column zero-padding retained), BCE-valid on all Train. Scalers use the entire allowed fit region. Seed 20261029, Adam 900 steps, batch 128, lr .001, L2 .0001, gradient clip 5. Invalid candidates have zero BCE weight; multiple correct candidates are positive labels.

BF/CF calibration uses exactly three internal Question folds, seed 20261130. Each of the three B/C/F score sets is predicted out of its fit region. Both 11-dimensional LME group-CE calibrators fit only the 2,114 disagreement OOF questions, using the original score SD, mean-map feature RMS, frequency/margin/group interactions, L2 .01, L-BFGS max300/gtol1e-7/ftol1e-10. Final base models are refitted on their full allowed populations. No in-sample final-model scores fit the calibrators. OOF/full-fit score-distribution shift remains a limitation.

Direct uses standardized 1024 embedding + 20 format/final inputs. Preprocessing fits all Train; optimization uses disagreement questions. It fits group CE with LME candidate pooling plus alpha*log(group size), weight L2 .01 and alpha-toward-1 L2 .01. Fixed MEAN initialization then LME, L-BFGS max300/gtol1e-6/ftol1e-10; final alpha 1.2377756304957468. It has no second calibrator.

`frozen_training.json` and source code define exact feature order and numeric operations. Canonical answers use the unchanged official parser. Invalid answers are excluded; ties select the earliest member, all-invalid inputs select 0, empty inputs return []. Base processing is float32; calibrator and Direct operations retain their frozen float64 paths.

Historical Train fivefold disagreement OOF on three partitions: 1736, 1733, 1737 / 2114. These historically exposed OOF results are not independent generalization evidence for the full-Train checkpoint. Existing official Validation is 746/798 overall, 303/355 disagreement. The subsequently authorized one-shot Official Test is 387/467 (82.86937901498929%); evaluation artifacts are in evaluation/test/. No model adjustment followed.

A prior fixed-protocol engineering reproduction reproduced all exported arrays bitwise in the pinned local environment (`records/training_reproduction.json`). Cross-platform bitwise equality is not guaranteed.
