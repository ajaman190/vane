# Models

## Version scheme

This repository package is **0.1.0** and does not ship weights. A model release is a semver. The first design target is **1.0.0**, status **unreleased**.

| Kind | Form | Example |
|------|------|---------|
| Release | semver | `1.0.0` |
| Versioned checkpoint id | `vane-<semver>-<modality>` | `vane-1.0.0-text`, `vane-1.0.0-vision` |
| Alias | stable pointer | `vane-text` → current stable text id; `vane-vision` → current stable vision id |

Until weights exist, aliases are **specified and unresolved**. A response reports the **versioned id that ran**, not the alias.

Same text-tower weights are shared across modalities. Vision adds a frozen vision encoder and a trained projection. A later semver may ship one modality without the other. Documentation is not split per version.

## Catalog (design target 1.0.0)

| Id / alias | Modality | Notes |
|------------|----------|-------|
| `vane-1.0.0-text` | text | Multilingual encoder, about 1B parameters. No vision tower. Context 32 768 tokens. |
| `vane-1.0.0-vision` | text + images | **Same** text tower. Frozen vision encoder; patch vectors projected into the token stream. Context 32 768 including image tokens. |
| `vane-text` | alias | Resolves to current stable text versioned id (unresolved until weights exist). |
| `vane-vision` | alias | Resolves to current stable vision versioned id (unresolved until weights exist). |

No audio checkpoint.

## Text checkpoint

- Multilingual encoder near 1B parameters.
- Windowed attention + latent bank as in [architecture.md](architecture.md).
- Option-set head for noul, choice, and score.
- Context limit: 32 768 tokens; longer requests are refused.

## Vision checkpoint

- Reuses the **same** text-tower weights as the text checkpoint of that release (when both ship).
- Frozen vision encoder.
- Trained projection of patch vectors into the token stream.
- Light update of the option-set head on image decisions; text tower is not retrained from scratch for images.
- Context limit: 32 768 tokens including image tokens.

## Router

The router does **not** run a network.

1. If the caller passed `model`, use that checkpoint (alias resolved to a versioned id).
2. Else if `state` contains one or more images, use `vane-vision`.
3. Else use `vane-text`.

Image detection is a **field check**, not a pixel inspection.

Cold load costs seconds. The field check costs microseconds. A server preloads the checkpoints it serves. If both are resident, they stay. If one slot, least-recently-used is dropped.

### Response routing fields

Every response includes:

| Field | Meaning |
|-------|---------|
| checkpoint | Versioned id that ran (e.g. `vane-1.0.0-text`) |
| reason | `explicit` \| `images` \| `text` |

- `explicit` — caller set `model`.
- `images` — no `model`; state had an images field.
- `text` — no `model`; no images field.
