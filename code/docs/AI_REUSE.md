# AI assistance and reuse disclosure

Codex assisted code implementation, controlled experiment execution, Full-Train conversion, packaging, documentation and automated checks during this project. Research directions and submission choices were made by the student and their coordinating chat; the student remains responsible for understanding and verifying the method.

This release reuses the already evaluated frozen Phase14 code and weights without retraining, parameter changes or new model selection. Official parser/runner/scorer code is reused unchanged and attributed in `THIRD_PARTY_NOTICES.md`. No external dataset, pretrained checkpoint, new LLM generation, network call, Reference input, ID lookup or cached prediction is used during inference. Train references are used only by the explicit training reproduction script; scoring reads references in a separate official process.

The final course report and the student's final authorship/assistance statement remain the author's responsibility.
