# Vane documentation

A vane points when the signal is steady and wobbles when it is not. Sharp \(q\) versus interior \(q\); that is what \(C(q)\) reports.

Vane is a decision model: one state plus a map of typed questions; one forward pass returns a probability for every allowed answer. Question ids travel with the answers and are not model inputs.

Package version `0.1.0`. Checkpoint ids and aliases: [models.md](models.md). Design targets `vane-1.0.0-text` and `vane-1.0.0-vision` are unreleased; aliases are unresolved until weights exist.

**Status.** Package `0.1.0`. Losses, the text trainer, eval, and a local server are in the tree. Published weights are not. A server with no checkpoint answers through `vane.model.untrained` (a small random network) so the schema is valid. That forward is not a trained model.

Configs: `src/vane/train/configs/`. Commands: `uv run vane-train`, `uv run vane-eval`, `uv run vane-serve`. The repository README has request and response examples.

## Doc map

| Doc | Contents |
|-----|----------|
| [architecture.md](architecture.md) | Request shape, primitives, confidence, network, windows, option-set head, score, cardinality temperature, early exit, languages, worked examples |
| [models.md](models.md) | Version scheme, text and vision checkpoints, router, catalog |
| [training.md](training.md) | Objective, losses, optimization order, calibration, release checks |
| [data.md](data.md) | Corpora, transforms, train / holdout / vision sets, splits |

Repository root README covers install, primitives, request examples, training, and the license.
