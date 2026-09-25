# Data

## Label policy

Human-labeled public sets. Soft labels \(\pi\) = vote share when multiple annotators exist. One-hot only for a certain fact or a single agreed label.

No model-written questions. No model-written labels.

**Not used as targets:** any corpus whose labels were written by a generative model. A second model’s distribution is not a target.

## Allowed transforms

Transforms of a human item only (no new target):

- paraphrase
- shuffle option order
- flip state between raw text and nested JSON
- inject a random distractor question

## Splits

Every set is split into **train**, **development**, **calibration**, and **test** **before** any checkpoint is selected.

| Split | Role |
|-------|------|
| train | Fit encoder (if unfrozen), option-set head, score components |
| development | Hyperparameters and monitoring during fit |
| calibration | \(T(\mathrm{type}, k)\), \(\tau(\mathrm{type}, k)\), and the check that a monotone map no longer reduces Brier |
| test | Opened once |

\(T(\mathrm{type}, k)\) and \(\tau(\mathrm{type}, k)\) are fit on the calibration split only. The isotonic gap is a training-batch term; the decision that the head is finished is read on the calibration split. The test split is opened once.

## Train text

| Corpus | Role | Primitive |
|--------|------|-----------|
| AmazonScience/massive | 51 languages, 60 intents | choice |
| clinc_oos | CLINC-150 in-scope plus out-of-scope | choice |
| Tobi-Bueck/customer-support-tickets | queue labels | choice |
| PolyAI/banking77 | 13 083 queries, 77 intents; **TRAIN split only** for fitting | choice, high \(k\) |
| google/boolq | yes/no | noul |
| nyu-mll/multi_nli | NLI | choice |
| xnli | cross-lingual NLI | choice |
| facebook/anli rounds 1–3 | adversarial NLI | choice |
| fever | supported / refuted / not-enough-info | choice |
| google-research-datasets/go_emotions | 28 emotions, multi-annotator | choice, soft \(\pi\) |
| google/civil_comments | toxicity | noul and score |
| deepset/prompt-injections | injection detect | noul |
| lmsys/toxic-chat | toxicity | noul |
| zefang-liu/phishing-email-dataset | phishing | noul |
| SetFit/enron_spam | spam | noul |
| nvidia/HelpSteer2 | five axes 0–4 | score |
| SetFit/sst5 | **TRAIN split only** | score |
| microsoft/ms_marco | binary passage relevance | noul |
| pfb30/multi_woz_2_2 | label = verified goal success at dialogue end; prefixes are states | noul and score |
| compiled knowledge quiz | short public facts; dated holdout whose answer exists only in a passage in the state | noul and choice |
| long-document probe | rare string at start / middle / end of a pad, and the pad without it; label = presence | noul |

### Explicit resolutions

- **SST-5:** train is in the mix so the ordinal head sees five levels. SST-5 test is **not** used to select checkpoints.
- **dair-ai/emotion:** entirely held out (not in training).
- **BANKING77:** train teaches \(k=77\). BANKING77 test is **not** used to select checkpoints.

## Held out (opened once)

- MASSIVE and XNLI languages not in the train split
- BANKING77 official test (3080)
- toxic-chat test
- prompt-injections test (\(n=116\))
- SST-5 test
- dair-ai/emotion entirely (not in training)
- a two-pole rubric written so both ends of a ticket are present
- the dated fact holdout

## Vision train (vision checkpoint only)

| Corpus | Role |
|--------|------|
| FUNSD | Form page image. Choice: field present or not; language of the page. |
| DocVQA | Page image plus a closed-label question. Answer is a short option list, not a generated span. |
| Rico screenshots | UI state. Choice of the next action from a short menu. |
