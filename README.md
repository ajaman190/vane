# Vane

Vane is a system-one model. It takes one state and a map of typed questions, and returns a probability for every allowed answer in one forward pass. It does not generate text.

A vane points when the signal is steady and wobbles when it is not. A sharp simplex is the steady case. An interior simplex is the wobble. \(C(q)\) reports which.

There are two checkpoints. `vane-text` reads text. `vane-vision` reads text and images. Language is not a route. Both use one multilingual text tower. Context is 32,768 tokens. Longer requests are refused.

HTTP is TypeSafe-compatible: `POST /v1/systemone` and `GET /v1/models`.

## Decision primitives

**Noul.** Yes or no. The answer is \(P(\mathrm{yes})\) in \([0, 1]\).

```json
{ "type": "noul", "noul": 0.91 }
```

**Choice.** Named options, from 2 to 255. `choice` is the name with the largest probability. `probabilities` is a simplex.

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": {"billing": 0.96, "technical": 0.02, "sales": 0.01, "other": 0.01},
  "confidence": 0.947
}
```

**Score.** An ordered rubric, from 2 to 10 levels, lowest first. `score` is the expectation \(\sum_i i\, q(i)\).

```json
{
  "type": "score",
  "score": 1.93,
  "legend": ["not urgent", "soon", "blocking"],
  "probabilities": {"0": 0.02, "1": 0.03, "2": 0.95},
  "confidence": 0.925
}
```

Question ids are chosen by the caller. They come back with the answers and are not model inputs.

## Example

Request:

```json
{
  "model": "vane-text",
  "state": {
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Refund the duplicate today or we cancel."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs and outages",
        "sales": "pricing and contracts",
        "other": "everything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["not urgent", "soon", "blocking"]
    },
    "refund": {
      "type": "noul",
      "instructions": "Does the user explicitly request a refund?"
    }
  }
}
```

Response shape:

```json
{
  "model": "vane-untrained-local-text",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.96, "technical": 0.02, "sales": 0.01, "other": 0.01},
      "confidence": 0.947
    },
    "urgency": {
      "type": "score",
      "score": 1.93,
      "legend": ["not urgent", "soon", "blocking"],
      "probabilities": {"0": 0.02, "1": 0.03, "2": 0.95},
      "confidence": 0.925
    },
    "refund": {"type": "noul", "noul": 0.97}
  },
  "usage": {"input_tokens": 40, "output_tokens": 0},
  "routing": {
    "checkpoint": "vane-untrained-local-text",
    "reason": "text"
  }
}
```

Add `"images"` on `state` and the same text is routed to the vision checkpoint.

Starter question maps, which you can edit: `vane.presets.triage_questions`, `guard_questions`, `moderation_questions`.

## Install

```bash
uv sync --extra serve
```

Training and eval also need the train extra:

```bash
uv sync --extra train --extra dev --extra serve
```

Core install does not require torch. Windows: `uv sync` in PowerShell from this directory works the same way. A plain venv is `pip install -e ".[serve]"`.

## Use

Local server, no API key:

```bash
uv run vane-serve --host 127.0.0.1 --port 8000
```

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"vane-text","state":"Refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the user request a refund?"}}}'
```

Python:

```python
from vane.client import Client

client = Client("http://127.0.0.1:8000")
result = client.systemone(
    model="vane-text",
    state="Refund the duplicate charge today or we cancel.",
    questions={
        "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
    },
)
print(result["answers"]["refund"]["noul"])
```

TypeScript: `@vane/sdk` in [`sdk/typescript/`](sdk/typescript/). No API key. Images are a field on the request.

TypeSafe clients can point here by setting `TYPESAFE_BASE_URL=http://127.0.0.1:8000` and `model` to `vane-text` or `vane-vision`. Their SDK will not send unless `TYPESAFE_API_KEY` is a non-empty string. This server ignores that header.

Docker, CPU by default:

```bash
docker compose up --build
```

GPU override: `docker compose -f compose.yaml -f compose.cuda.yaml up --build`. The image does not download weights.

## Train and eval

Configs are `src/vane/train/configs/base.yaml`, `vane-text-1.0.0.yaml`, and `vane-vision-1.0.0.yaml`. Checkpoints are written under `src/vane/train/checkpoints/`.

`vane-text-1.0.0.yaml` and `vane-vision-1.0.0.yaml` are full runs: one epoch, no row cap, CUDA, frozen `jhu-clsp/mmBERT-base` and its tokenizer. The vision file also loads frozen SigLIP2 and keeps the text corpora so the head does not forget text questions. The hash tokenizer is not a training path.

```bash
uv sync --extra train
uv run vane-train --config src/vane/train/configs/vane-text-1.0.0.yaml
uv run vane-train --config src/vane/train/configs/vane-vision-1.0.0.yaml
```

The text config streams a capped number of rows from each corpus. The calibration slice is not used as training data. After the steps, a scalar temperature per primitive is written beside the checkpoint. The training loss is still log loss plus Brier on the simplex. Details are in [docs/training.md](docs/training.md).

Load a saved directory or a Hub repo:

```python
from vane import load

checkpoint = load("/path/to/checkpoint-dir", modality="text")
```

The directory needs `config.json` and `model.safetensors`. A Hub id downloads only the requested modality (`uv sync --extra hub`).

## Citation

If you use Vane in your research, please cite it.
```bibtex
@software{vane2026,
  title   = {Vane: An Open-Source System One Decision Model},
  year    = {2026},
  author  = {ajaman190},
  url     = {https://github.com/ajaman190/vane},
  license = {Apache-2.0}
}
```

## License

[Apache License 2.0](LICENSE).
