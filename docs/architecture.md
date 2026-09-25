# Architecture

## What Vane is

A decision model. The caller supplies one **state** and a **map of typed questions**. One forward pass returns a probability for every allowed answer of every question. Nothing is decoded to text. Question map keys are chosen by the caller; they are returned with the answers and are not fed into the encoder.

All questions in one request share one encoding of the state.

## Request

| Field | Type | Notes |
|-------|------|-------|
| `state` | string, JSON object, list of strings, or any of those plus an `images` array of image bytes | Encoded once per request |
| `questions` | map | Keys are caller-chosen ids |
| `model` | optional string | `vane-text`, `vane-vision`, a versioned checkpoint id, or omitted (router chooses) |

## Primitives

### Noul

Yes / no. Answer shape:

```json
{ "type": "noul", "noul": q }
```

where \(q \in [0,1] = P(\mathrm{yes})\). No confidence field.

### Choice

Named options, \(2 \leq k \leq 255\). Each option has a description (possibly null). Answer shape:

```json
{ "type": "choice", "choice": "<argmax name>", "probabilities": q, "confidence": C }
```

\(q\) is a simplex over the option names. `choice` is \(\arg\max_i q_i\). Confidence is \(C(q)\).

### Score

Ordered rubric with \(L\) levels, \(2 \leq L \leq 10\), lowest first. Answer shape:

```json
{ "type": "score", "score": E, "legend": [...], "probabilities": q, "confidence": C }
```

where

\[
E = \sum_i i\, q(i).
\]

\(q\) is a simplex over levels. Confidence is \(C(q)\).

## Confidence

For a simplex \(q\) with \(k \geq 2\) support points:

\[
C(q) = \frac{k \cdot \max_i q_i - 1}{k - 1}.
\]

On a simplex, \(\max_i q_i \in [1/k, 1]\). \(C = 1\) iff \(q\) is a point mass. \(C = 0\) iff \(q\) is uniform.

\(C\) is **serve-only**. It does not take the label. It is not a training objective. It is computed after temperature scaling, on the simplex returned to the caller. Noul does not use \(C\).

## Network

```
state text
  and, on vane-vision only, image patches
  from a frozen vision encoder, projected into the token stream
        │
        ▼
pack to n tokens                         refuse if n > 32,768
        │
        ▼
windows of 128                           bidirectional attention inside the window
        │                                a token is fully visible before the summary
        ▼
one latent per window                    the latent cross-attends into its own window
        │
        ▼
latent bank                              the c latents attend to each other
        │                                every question in the request reuses this bank
        ▼
early copy of the option-set head
        │
        ├─ every question has C(q) ≥ τ     stop and return those simplices
        └─ any question is still interior  run the remaining blocks, then the full head
        ▼
option-set head                          each option is a vector, cross-attended to the bank
                                         two transformer layers, no position ids
                                         permuting the options permutes the logits only
        │
        ├── noul      one logit, sigmoid. The wire field is P(yes). No confidence.
        ├── choice    softmax(z / T(choice, k)), then C(q)
        └── score     two CORN chains
                      μ₂ = μ₁ + softplus(δ)
                      q = α P(lower) + (1−α) P(higher)
                      the reported score is E[Y], then C(q)
```

\(T(\mathrm{type}, k) = \mathrm{softplus}(a_{\mathrm{type}} + b_{\mathrm{type}} \log k)\) is applied to the logits before the simplex. \(C(q)\) is computed on that simplex and is not a loss. There is no audio path. Counts, spans, and tallies stay in the caller.

Pipeline in words:

1. Text tokens. On the vision checkpoint, frozen vision patches are projected into the same stream and count against the 32,768 budget.
2. Windows of width 128. Full attention inside a window. No \(n \times n\) layer.
3. One latent per window, then a bank where those latents attend to each other.
4. An early copy of the option-set head. Later blocks run only if some question is still interior.
5. The full option-set head. Noul is a sigmoid. Choice is a softmax. Score is the two-component ordinal mixture, reported as an expectation.

## Windows

Window width \(w = 128\). Packed length \(n\) is padded so \(n\) is a multiple of \(w\). Number of windows:

\[
c = \frac{n}{w}.
\]

Pair counts for one layer (one pair per query–key; no head factor; no MLP):

| Term | Count |
|------|-------|
| Inside-window attention | \(c \cdot w^2 = n w\) |
| Latent bank | \(c^2\) |
| Full \(n \times n\) (not used) | \(n^2\) |

Cross-attention from each latent into its own window adds \(n\) pairs and does not change the asymptotic ratio of windowed cost to full attention.

| \(n\) | \(n^2\) | windows (\(nw + c^2\)) | ratio \(n^2 / (nw + c^2)\) |
|------:|--------:|-------------------------:|---------------------------:|
| 4096 | 16 777 216 | 525 312 | \(31.9\times\) |
| 32 768 | 1 073 741 824 | 4 259 840 | \(252\times\) |

There is no \(n \times n\) attention layer. Requests longer than 32 768 tokens are **refused**, not cropped.

A token is fully visible inside its window before the window is summarized. If a held-out rare-string probe fails, widen \(w\). Do not add an \(n \times n\) layer to cover a failed probe.

Cache for another question on the same state: the \(c\) latents.

## Option-set head

The option and question vectors cross-attend two memories. The latent bank is what a later question reuses. The token states are what a rare string needs: one query against every token, which is linear in the length, while token-to-token attention stays inside the window.

\[
q = \mathrm{softmax}\!\left(\frac{z}{T(\mathrm{type}, k)}\right).
\]

Noul is one logit through a sigmoid. It is not tied to a choice unless the schema marks the pair as complements. Two unmarked questions about a statement and its negation are not forced to sum to 1.

## Score (CORN, two components)

One ordinal chain:

\[
\begin{aligned}
f_1 &= P(Y > 0), \\
f_r &= P(Y > r-1 \mid Y > r-2), \\
P(Y > r-1) &= \prod_{j=1}^{r} f_j.
\end{aligned}
\]

Mass on each level is the difference of the survival function. One chain is unimodal.

Two chains, with the second location forced above the first:

\[
\mu_2 = \mu_1 + \mathrm{softplus}(\delta),
\]

\[
q = \alpha\, P(Y; \mu_1) + (1-\alpha)\, P(Y; \mu_2).
\]

\(\alpha\) weights the **lower** component. Softplus stops the optimizer from swapping component names.

\[
\mathbb{E}[Y] = \sum_j j\, q(j) = \sum_{i=0}^{L-2} P(Y > i).
\]

A single softmax can dump mass on the middle level when the state has two poles. Two components can put mass on both ends.

## Cardinality temperature

Planned map:

\[
T(\mathrm{type}, k) = \mathrm{softplus}(a_{\mathrm{type}} + b_{\mathrm{type}} \log k).
\]

One \((a, b)\) pair per primitive. \(C\) is computed after this division. The shipped trainer does **not** fit \(a\) and \(b\); it grid-searches a scalar \(T\) per primitive and writes `temperature.json`. Fitting and freeze order: [training.md](training.md).

## Early exit

Early exit is a **skip of later encoder blocks**. It is not a skip of a question, not a smaller model, not the router, and not an “I don’t know” answer.

After block 1, a cheap copy of the option-set head writes a simplex per question. If **every** simplex has \(C(q) \geq \tau(\mathrm{type}, k)\), remaining encoder blocks do not run. If **any** question is interior, the rest of the stack runs and the head is applied again. One uncertain question keeps the stack on for the whole request. Per-question encoder skip would encode the state twice. During training, a coin decides whether a step stops after block 1 and still takes the proper-score loss; serve time uses \(\tau\), not the coin. \(\tau\) is not fit yet. Details: [training.md](training.md).

Two-pole scores have low \(C\) even when the poles are sure, so they do not exit. A 77-way choice with \(\max q = 0.2\) has \(C \approx 0.19\) and does not exit.

Expected compute cost:

\[
\mathbb{E}[\mathrm{cost}] = f \cdot \mathrm{cost}_1 + (1-f) \cdot \mathrm{cost}_{\mathrm{full}}.
\]

\(f\) is not measured. Do not publish a speedup.

An interior simplex is still returned, with low \(C\).

## Languages

Both checkpoints are multilingual. Language is not a route.

The vocabulary covers Latin, CJK, Devanagari, Hangul, Arabic, and other scripts in the training mix. A cheap script-histogram token may be prepended as a feature, not as a switch between weights.

Do not use an English-only vocabulary: a tower that cannot read a script still returns a sharp simplex, and a confidence gate then acts on a guess.

Equal accuracy across languages is not promised. Thin slices should spread the simplex rather than collapse to a point mass. That is an evaluation item.

## Worked examples

Probabilities below are **illustrative shapes of the simplex**, not measured results.

### 1. English JSON refund ticket (early exit)

State: English JSON support ticket about a refund. Model: `vane-text`. Four questions; early exit fires.

| Question | Type | Illustrative output |
|----------|------|---------------------|
| department | choice (\(k=4\)), argmax billing | \(\max q = 0.96\) → \(C = (4\cdot 0.96 - 1)/3 = 0.947\) |
| urgency | score (\(L=3\)) | \(E \approx 1.93\), \(C \approx 0.925\) |
| refund | noul | \(q = 0.97\) |
| phishing | noul | \(q = 0.01\) |

Every choice/score \(C\) clears \(\tau\), so later encoder blocks are skipped.

### 2. Hindi Devanagari (same questions)

Same question map; state in Hindi (Devanagari). Still `vane-text`. A script-histogram token is prepended. The simplex is wider than the English case; example billing \(\max q = 0.78\) → \(C = (4\cdot 0.78 - 1)/3 = 0.707\); refund noul \(0.88\). Whether early exit fires depends on \(\tau(\mathrm{type}, k)\).

### 3. Two-pole ticket

Calm opening plus furious close. Score over three levels with illustrative mass

\[
q \approx (0.41,\ 0.08,\ 0.51),\qquad
E = 0\cdot 0.41 + 1\cdot 0.08 + 2\cdot 0.51 = 1.10,\qquad
C = \frac{3\cdot 0.51 - 1}{2} = 0.265.
\]

Noul `payroll-blocked` \(\approx 0.96\). No early exit: the score \(C\) is low. Argmax is the furious end; the mean is not the middle level (middle mass stays small).

### 4. Invoice image

State includes an invoice page image → router selects `vane-vision`. Illustrative questions: `signed` (noul), `lang` (choice), `total_present` (noul).

### 5. Long pad with a rare codeword

Pad of \(n = 16\,384\) tokens; codeword at token 8000. Windows: \(c = 128\).

\[
nw + c^2 = 2\,097\,152 + 16\,384 = 2\,113\,536
\quad\text{vs}\quad
n^2 = 268\,435\,456.
\]

Presence noul near \(0.99\). A fact not in the pad near \(0.5\). The interior fact blocks early exit for the whole request.
