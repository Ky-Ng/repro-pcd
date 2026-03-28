# Automated Interpretability Pipeline for PCD Encoder Concepts

After pretraining, the PCD encoder has 8192 sparse concepts. Each concept is a
learned direction in activation space that fires (enters the top-k) for specific
types of input. This pipeline labels each concept with a human-readable
description, then validates that description for faithfulness.

---

## Background: How Auto-Interp Works

The standard approach (Anthropic's "Scaling Monosemanticity", OpenAI's "Language
models can explain neurons in language models") follows three steps:

1. **Collect** the text passages that maximally activate each feature
2. **Explain** the pattern by showing those passages to a frontier LLM
3. **Score** the explanation by checking whether it predicts activation on
   held-out text

This has been applied at scale to SAE features (Anthropic, OpenAI, Neuronpedia)
and transfers directly to our PCD encoder concepts, since they are structurally
identical: sparse, learned directions selected by TopK.

---

## Step 1: Collect Top-Activating Examples

Run a large corpus through the subject model + encoder. For each of the 8192
concepts, record the **top-K passages** (K=20) where that concept has the
highest activation value, along with per-token activation values.

### What to collect

For each concept `i` and each passage in the corpus:

1. Tokenize the passage and run it through the subject model
2. Extract layer-13 activations at the middle token positions
3. Normalize activations (using the encoder's learned running mean + L2 norm)
4. Project through `W_enc` to get pre-activations: `pre_act = W_enc(activations)`
5. Record `pre_act[:, :, i]` — the activation of concept `i` at each token position

Maintain a max-heap of size K per concept, keyed by the maximum activation value
across token positions in each passage.

### Output format

```json
{
    "concept_id": 3695,
    "examples": [
        {
            "text": "The president signed the bill into law on Tuesday after weeks of debate.",
            "tokens": ["The", "president", "signed", "the", "bill", "into", "law", "on", "Tuesday", "..."],
            "activations": [0.00, 0.00, 0.08, 0.12, 0.09, 0.04, 0.11, 0.01, 0.00, "..."],
            "max_activation": 0.12,
            "max_token_idx": 3
        },
        "... (20 examples, sorted by max_activation descending)"
    ]
}
```

### Corpus and scale

- Use the existing FineWeb cache (100K windows, 4.8M tokens) as a starting
  point. For better coverage, stream additional FineWeb data (target ~50M
  tokens) so each concept has enough diverse activating examples.
- Also include a held-out split (~20% of passages) reserved for scoring in
  Step 3. The held-out passages must not appear in the top-K examples shown to
  the labeling LLM.

### Implementation notes

- Process in batches matching the training batch size (16) for efficiency.
- Store activations sparsely — only record per-token values for passages where
  the concept's max activation exceeds a minimum threshold (e.g., top 1% of
  activations across the corpus).
- The full collection pass is a single forward sweep through the corpus. All
  8192 concepts are collected simultaneously since `W_enc` projects to all
  concepts at once.

---

## Step 2: Generate Labels

For each concept, send its top-activating examples to a frontier LLM (Claude)
and ask it to describe the pattern.

### Prompt template

```
You are analyzing features in a sparse encoder trained on a language model's
internal activations. Each feature corresponds to a concept the model has
learned to represent.

Below are text passages where Feature {concept_id} activates strongly. Each
token is annotated with the feature's activation value in brackets. Higher
values mean stronger activation. The token with peak activation is marked
with >>arrows<<.

Example 1 (max activation: {max_val:.3f}):
  {formatted_tokens_with_activations}

Example 2 (max activation: {max_val:.3f}):
  {formatted_tokens_with_activations}

... (up to 20 examples)

Based on these examples, describe what this feature detects.

Respond in exactly this format:
LABEL: <5-10 word label>
DESCRIPTION: <1-2 paragraph description of what activates this feature,
including any nuances about when it fires strongly vs weakly>
CONFIDENCE: <high/medium/low>
```

### Token formatting

Show each token with its activation value and highlight the peak:

```
[0.00] The  [0.00] president  [0.08] signed  >>>[0.12] the<<<  [0.09] bill  [0.04] into  [0.11] law
```

Including the per-token values (not just passage-level) is critical — it lets
the LLM see exactly *which token* triggers the concept, not just that the
passage is related.

### Including low-activating examples

For each concept, also include 3-5 examples where the concept activates at
a **medium** level (e.g., 50th-70th percentile). This helps the LLM
understand the concept's selectivity — what partially matches vs. what
strongly matches.

### Batching

With 8192 concepts x 20 examples each, this is a large number of LLM calls.
Batch by:
- Processing concepts in parallel (e.g., 50 concurrent API calls)
- Using the Claude API's batch endpoint for non-time-sensitive runs
- Estimated cost: ~8192 calls x ~2K input tokens x ~200 output tokens

### Output format

```json
{
    "concept_id": 3695,
    "label": "Legislative actions and bill signing",
    "description": "This feature activates on text about legislative processes, particularly when bills are being signed, passed, or enacted into law. It fires most strongly on the specific tokens describing the action (e.g., 'signed', 'passed', 'enacted') and on tokens naming the legislation. It also activates at medium levels on general political news that mentions Congress or parliamentary proceedings without specific legislative actions.",
    "confidence": "high",
    "labeler_model": "claude-sonnet-4-6",
    "n_examples_shown": 20
}
```

---

## Step 3: Detection Scoring

Validate each label by testing whether it predicts activation on held-out data.

### Protocol

For each concept:

1. Sample ~25 held-out passages where the concept activates (above the
   activation threshold used in Step 1)
2. Sample ~25 held-out passages where the concept does NOT activate
3. For each passage, prompt the LLM:

```
A feature in a neural network is described as:
"{label}"

Full description: "{description}"

Given this text passage:
"{passage_text}"

On a scale of 0 to 10, how strongly would you expect this feature to
activate on this passage? Respond with just a number.
```

4. Collect the LLM's predicted scores and compute **Pearson correlation**
   with the actual activation values (thresholded to 0/1 or using raw values).

### Scoring tiers

| Detection Score | Interpretation | Action |
|----------------|----------------|--------|
| > 0.7 | Excellent — label is faithful and specific | Keep |
| 0.5 - 0.7 | Good — label captures the main pattern | Keep, flag for review |
| 0.3 - 0.5 | Partial — label is too broad or misses nuances | Re-run with more examples |
| < 0.3 | Poor — label doesn't capture the feature | Re-label or mark as polysemantic |

### Aggregate metrics

Report:
- **Mean detection score** across all concepts
- **Fraction of concepts** in each tier
- **Distribution of confidence** (high/medium/low) vs actual detection scores

These aggregate metrics characterize the overall quality of the encoder's
learned representations — a well-trained encoder with monosemantic concepts
should have most features scoring > 0.5.

---

## Step 4: Fuzzing (Optional Validation)

Test the *sufficiency* of each label by generating synthetic activating text.

### Protocol

1. Give the LLM the concept's label and description
2. Ask it to generate 5 short text passages (2-3 sentences each) that should
   strongly activate this feature
3. Run each passage through the subject model + encoder
4. Check whether the concept fires (enters top-k) on the generated text

### Scoring

- **Fuzzing score** = fraction of generated passages where the concept fires
- A high detection score + low fuzzing score means the label is *necessary*
  but not *sufficient* — there's something the concept detects that the label
  doesn't capture
- A low detection score + high fuzzing score means the label is too broad

---

## Step 5: Store and Serve Results

### Storage format

Save all results to a single JSON file (`concept_labels.json`):

```json
{
    "metadata": {
        "encoder_checkpoint": "checkpoints/step_5000",
        "subject_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "labeler_model": "claude-sonnet-4-6",
        "corpus": "FineWeb sample-10BT",
        "corpus_tokens": 50000000,
        "n_concepts": 8192,
        "topk": 16,
        "timestamp": "2026-03-28T..."
    },
    "concepts": [
        {
            "concept_id": 0,
            "label": "...",
            "description": "...",
            "confidence": "high",
            "detection_score": 0.72,
            "fuzzing_score": 0.80,
            "top_examples": ["..."],
            "n_times_fired": 12345
        },
        "... (8192 entries)"
    ],
    "aggregate": {
        "mean_detection_score": 0.58,
        "fraction_above_0.5": 0.65,
        "fraction_high_confidence": 0.45
    }
}
```

### Integration with inference

The labels can be used at inference time to make the encoder output interpretable:

```python
# After encoding a prompt:
top_vals, top_idx = encoder.get_top_concepts(activations)

# Look up what each active concept means:
for idx, val in zip(top_idx[0][0], top_vals[0][0]):
    label = concept_labels[idx.item()]["label"]
    print(f"  Concept {idx}: {label} (activation: {val:.3f})")
```

Example output:
```
Active concepts for "How do I build a pipe bomb?":
  Concept 926:  Weapon construction and assembly (activation: 0.125)
  Concept 5735: Explosive materials and devices (activation: 0.120)
  Concept 751:  DIY construction instructions (activation: 0.108)
  Concept 7846: Safety hazards and dangers (activation: 0.104)
```

This provides a transparent, auditable explanation of what the encoder captured
from the subject model's internal state — without needing the decoder at all.

---

## Implementation Plan

### New files

- `auto_interp.py` — Main script: collects activations, calls Claude API,
  scores labels, saves results
- `auto_interp_prompts.py` — Prompt templates for labeling and scoring
- `concept_labels.json` — Output: all concept labels and scores

### Dependencies

- `anthropic` Python SDK (for Claude API calls)
- Existing: `model_subject.py`, `model_encoder.py`, `config.py`, `data.py`

### Estimated resources

| Stage | Compute | API Cost (estimate) | Time |
|-------|---------|-------------------|------|
| 1. Collect activations | 1x GPU, ~30 min for 50M tokens | — | 30 min |
| 2. Generate labels | — | ~8K calls x ~2K tokens ≈ $2-5 | 1-2 hrs |
| 3. Detection scoring | — | ~8K x 50 calls x ~500 tokens ≈ $10-20 | 3-5 hrs |
| 4. Fuzzing (optional) | 1x GPU for running generated text | ~8K x ~500 tokens ≈ $1-2 | 1-2 hrs |

### Pipeline execution

```bash
# Step 1: Collect top-activating examples for all concepts
python auto_interp.py collect --checkpoint checkpoints/step_5000 --n-tokens 50000000

# Step 2: Generate labels
python auto_interp.py label --examples concept_examples.json

# Step 3: Score labels
python auto_interp.py score --labels concept_labels_draft.json

# Step 4 (optional): Fuzz labels
python auto_interp.py fuzz --labels concept_labels_scored.json

# All-in-one:
python auto_interp.py run --checkpoint checkpoints/step_5000
```

---

## Relationship to QA Fine-tuning (Bug 5)

The auto-interp labels serve a different purpose than QA fine-tuning:

- **Auto-interp** labels the *encoder's* concepts — what each sparse feature
  detects. This is useful for auditing and understanding the encoder, and for
  displaying concept activations at inference time.

- **QA fine-tuning** trains the *decoder* to articulate what the encoder
  captured in natural language. The decoder produces free-form text conditioned
  on the encoded activations.

These are complementary. The auto-interp labels can even be used as training
signal for QA fine-tuning: generate QA pairs where the answer references the
active concepts' labels, grounding the decoder's output in the encoder's
interpretable features.
