# Sources and reuse

- Official assignment and byte-identical numeric parser, prediction runner and independent scorer: https://github.com/xudongwu-0/DASE7506-MP2. Protocol: `dase7506-mp2-open-hard-verifier-v5`. Copies in `official/` preserve behavior; no official data are bundled.
- Questions/references originate from OpenAI GSM8K under its retained MIT notice. Course candidate traces/embeddings were generated with Qwen3-0.6B; full revision and attribution are in `official/DATA_NOTICE.txt`. No external pretrained weights are loaded by this method.
- Project-local prior student implementations supplied the frozen embedding scorers, format/final features, group calibration and Direct-LME training. Their code and fitted inference arrays are retained byte-for-byte from the evaluated Phase14 package; algorithm hashes appear in `VERSION.json` and `evaluation/validation_provenance.json`.
- Runtime dependencies: NumPy, SciPy and threadpoolctl. Their own licenses apply to the installed packages; dependencies are installed separately and not redistributed in this repository.
- This package does not assign a license on behalf of upstream authors. See `LICENSE_STATUS.md` before public release.
