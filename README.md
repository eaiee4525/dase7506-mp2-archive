# DASE7506 Mini Project 2

Frozen Phase14: **BF25% + CF25% + Direct-LME50%**. Official Test: **387/467**, scorer accuracy **82.86937901498929%**.

- [Project guide](GUIDE.md)
- [Installation, training, prediction and official scoring commands](code/README.md)
- [Final Report (9 pages)](report/DASE7506_MP2_report.pdf)
- [Inference checkpoint/configuration](code/assets/)
- [Checkpoint download](https://github.com/eaiee4525/dase7506-mp2/releases/download/v1.0.0/phase14_checkpoint_v1.zip)
- [Official Test JSON](results/evaluation/test/score.json)
- [Experiment evidence](results/experiments/INDEX.json)
- [Resource and interface evidence](results/resource/)
- [AI/reuse disclosure](code/docs/AI_REUSE.md), [sources](code/docs/THIRD_PARTY_NOTICES.md), [license notices](code/docs/LICENSE.md)

## Layout

```text
code/       Frozen training/inference code, dependencies, configuration and checkpoint
report/     Author-approved final PDF
results/    Official evaluation, experiment and resource records
```

From the repository root, enter `code/` before following the complete reproduction commands:

```sh
cd code
```

The checkpoint ZIP contains an `assets/` directory and should be extracted inside `code/` when downloaded separately. All assets are already included in this repository.

Inference assets:61,839 bytes uncompressed. Local complete Test prediction+scoring:0.640seconds; reference Xeon eligibility has not been measured. See the detailed resource limitations in the running instructions. No dataset or reference answers are included.
