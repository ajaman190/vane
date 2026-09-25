# Training

## Objective

The trained object is the returned simplex \(q\), or \(P(\mathrm{yes})\) for a noul. \(q\) is trained to match the outcome distribution.

**Direct proper scoring.** Differentiate the loss through \(q\). No sampled action, no noise on logits, no policy gradient, no group baseline.

Target \(\pi\): annotator vote share when annotators disagree; a point mass when the label is certain. A one-hot on an uncertain item is not used. No model-written labels as targets; do not use a single external teacher’s distribution as the target. Corpora, holdouts (SST-5 test, dair-ai/emotion, BANKING77 test), and the train / development / calibration / test split order: [data.md](data.md).

Log loss and Brier are each strictly proper. Their sum is strictly proper. The sum reweights the same minimizer; it does not add a second one.

Arithmetic, exact counts, and string spans are computed by the caller. They are not questions.

## Losses

### Noul

\[
L_{\mathrm{noul}} = -\bigl(\pi \log q + (1-\pi)\log(1-q)\bigr) + (q - \pi)^2.
\]

### Choice

\[
L_{\mathrm{choice}} = -\sum_j \pi_j \log q_j + \|q - \pi\|_2^2.
\]

### Score

Ranked probability score:

\[
\mathrm{RPS}(q,\pi) = \frac{1}{L-1} \sum_{i=0}^{L-2} \bigl(F_q(i) - F_\pi(i)\bigr)^2.
\]

\[
L_{\mathrm{score}} = -\sum_j \pi_j \log q_j + \|q - \pi\|^2 + \beta \cdot \mathbf{1}_{\mathrm{unimodal}} \cdot \mathrm{RPS} + \gamma\, H(\alpha).
\]

RPS is applied only when \(\pi\) is unimodal. The entropy penalty \(H(\alpha)\) on the gate \(\alpha\) keeps the second ordinal component off unless the likelihood needs it.

## Isotonic residual

On a batch of at least 64 scalar predictions:

1. Fit an isotonic map from predicted probability to outcome.
2. Detach that map.
3. Add the positive part of the **batch-mean** Brier gap between the predictions and the mapped predictions.

If a monotone recalibration still helps, the head is not finished. The positive part wraps the batch mean, not each row. The gap is a term on training batches. Whether the head is finished is judged on the calibration split, not on the batches that trained it.

## Cardinality temperature and early-exit thresholds

The forward pass uses

\[
T(\mathrm{type}, k) = \mathrm{softplus}(a_{\mathrm{type}} + b_{\mathrm{type}} \log k).
\]

Those \(a\) and \(b\) are fit last, on the calibration split only, then frozen. The trainer now runs that fit for a few steps with every other weight frozen, and writes the values under `cardinality` in `temperature.json`. The scalar grid search remains in the same file as a record. It is not a substitute for \(a\) and \(b\). \(\tau\) is not fit yet.

## Optimization order

1. Start from a published multilingual encoder near 1B parameters. Do not pretrain from scratch.
2. Freeze it. Fit the option-set head and score components on the text checkpoint.
3. Unfreeze the encoder at a small learning rate only if a held-out fact quiz is still weak.
4. Fit \(T(\mathrm{type}, k)\) and \(\tau(\mathrm{type}, k)\) on the calibration split only.
5. Build the vision checkpoint from the same text tower. Frozen vision encoder. Train only the projection into the token stream, plus a light update of the head on image decisions. Do not retrain the text tower from scratch for images.

**Early-head training.** A random fraction of steps stop after block 1 and still take the proper-score loss. Serve time uses \(\tau\), not the coin.

## Release checks

No measured accuracy numbers are claimed here. Checks to run before a release:

1. **Rare-string probe** at 4k and 17k tokens, three positions (start, middle, end), and the pad with the string removed.
2. **Two-pole rubric:** mass on both ends; expectation may sit in the middle while the middle level stays small.
3. **Fact quiz** with and without a passage in the state.
4. **MASSIVE** and **XNLI** on English and at least ten other languages: accuracy and ECE. A language near chance must not sit at \(C\) near 1.
5. **Choice** \(k=3\) and \(k=77\). Check \(C\) by hand against the formula.
6. **Router:** text request → text checkpoint; same text plus an image → vision; `model` overrides both.
7. **Warm latency** on one GPU, text and text-plus-one-image, at 512 and 8k tokens. The pair-count table in [architecture.md](architecture.md) is not a latency table.
