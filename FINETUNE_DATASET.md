# Fine-tuning Dataset Generation

How the QA fine-tuning data was generated for Stage 2 of the PCD pipeline.

---

## Overview

The paper (arXiv:2512.15712) fine-tunes on **SynthSys(8B)**, a dataset of QA
pairs about user attributes inferred by the subject model (e.g., "What gender
does the model assume the user is?"). Since SynthSys is not publicly available,
we constructed a substitute dataset from FineWeb text passages using the subject
model as a self-classifier.

**Script:** `data_finetune.py`
**Cache:** `data_cache/qa_pairs.json`
**Size:** ~10,000 QA pairs

---

## Step-by-Step Generation Process

### Step 1: Load Tokenized FineWeb Windows

Reuse the same pre-tokenized 48-token windows from the pretraining cache
(`data_cache/fineweb_windows.pt`). Each window is a contiguous chunk of web
text split into:

```
[16 prefix tokens] [16 middle tokens] [16 suffix tokens]
```

Only `prefix_ids` and `middle_ids` are used for QA generation. The suffix is
discarded (it was only needed for the pretraining next-token prediction
objective).

### Step 2: Decode Window to Text

Each 32-token window (`prefix + middle`) is decoded back to a text string
using the tokenizer. Windows shorter than 30 characters are skipped.

```python
text = tokenizer.decode(window.tolist(), skip_special_tokens=True)
```

### Step 3: Select a Random Question Template

One of 8 attribute templates is chosen at random. Each template probes a
different text property:

| Template Field | Example Question |
|---------------|-----------------|
| `topic` | "What is the primary topic of the text? A. {a} B. {b} C. {c} D. {d}" |
| `sentiment` | "What is the tone or sentiment? A. Positive B. Negative C. Neutral D. Mixed" |
| `domain` | "What domain does the text belong to? A. {a} B. {b} C. {c} D. {d}" |
| `formality` | "Is the text formal or informal? A. Formal B. Informal C. Semi-formal D. Cannot determine" |
| `content_type` | "What type of content is this? A. {a} B. {b} C. {c} D. {d}" |
| `about` | "What is the text primarily about? A. {a} B. {b} C. {c} D. {d}" |
| `factuality` | "Does the text contain factual claims or opinions? A. Mostly factual B. Mostly opinions C. Mix D. Neither" |
| `audience` | "What is the intended audience? A. {a} B. {b} C. {c} D. {d}" |

Templates with `{a}`, `{b}`, `{c}`, `{d}` placeholders have their options
filled from category pools (see Step 4).

### Step 4: Classify the Passage Using the Subject Model

The subject model (Qwen2.5-1.5B-Instruct, the same frozen model used in the
PCD pipeline) is prompted to classify the passage. The prompt format depends
on the template field:

**For pool-based fields** (topic, domain, content_type, audience):

```
Classify the topic of this text in one word: "first 200 chars of passage"
Topic:
```

The subject model generates up to 5 tokens. The output is fuzzy-matched
against a predefined pool of candidates. For example, the topic pool is:

```python
["science", "technology", "politics", "sports", "entertainment",
 "health", "education", "business", "travel", "food", "history",
 "nature", "art", "music", "literature", "philosophy", ...]
```

If the model's output contains any pool entry (e.g., generates "This is about
technology and..."), that entry becomes the correct answer ("technology"). If
no match is found, a random pool entry is selected as fallback.

**For fixed-option fields** (sentiment, formality, factuality):

```
Text: "first 200 chars of passage"
What is the tone or sentiment of the text?
A. Positive
B. Negative
C. Neutral
D. Mixed
Answer:
```

The model generates up to 3 tokens. The first letter (A/B/C/D) found in the
output becomes the answer.

**For the "about" field:**

```
What is this text about in 2-3 words? "first 200 chars of passage"
About:
```

The model generates up to 8 tokens, truncated to the first phrase. This
becomes one of the four MC options alongside 3 random topic-pool distractors.

### Step 5: Construct Multiple-Choice Options

For pool-based fields, the correct answer is combined with 3 random
distractors from the same pool (ensuring no duplicates). The 4 options are
shuffled into random A/B/C/D positions.

```python
correct = "technology"
distractors = random.sample([t for t in TOPIC_POOL if t != correct], 3)
# e.g., ["sports", "history", "art"]
options = [correct] + distractors
random.shuffle(options)
# e.g., ["history", "technology", "art", "sports"]
correct_letter = "B"  # technology is at index 1
```

### Step 6: Store the QA Pair

Each example is stored as a dict:

```python
{
    "prefix_ids": [16 token IDs],     # context tokens
    "middle_ids": [16 token IDs],     # encoder input tokens
    "question": "What is the primary topic of the text?\nA. history\nB. technology\nC. art\nD. sports\nAnswer:",
    "answer": " B",                   # single letter, space-prefixed
}
```

### Step 7: Tokenize for Training

The `QADataset.__getitem__` method tokenizes the question and answer strings
into fixed-length ID tensors:

- `question_ids`: padded/truncated to `max_q_len=80` tokens
- `answer_ids`: padded/truncated to `max_a_len=8` tokens

Padding uses the tokenizer's pad token ID.

### Step 8: Cache to Disk

All generated pairs are serialized to `data_cache/qa_pairs.json` (tensors
converted to lists). Subsequent runs load from cache without re-running the
subject model.

---

## How QA Data Is Used During Training

During fine-tuning (`train_finetune.py`), each training step randomly selects
either a QA batch or a FineWeb batch (50/50 mix):

**QA batch:**
1. Feed `[prefix_ids + middle_ids]` through the frozen subject model → activations
2. Feed activations through the frozen encoder → soft tokens `[B, 16, 1536]`
3. Decoder sees `[soft_tokens] + [question_embeds]` → predicts `[answer_ids]`
4. Cross-entropy loss on answer tokens only (question positions excluded)

**FineWeb batch** (identical to pretraining):
1. Feed `[prefix_ids + middle_ids]` through frozen subject model → activations
2. Feed activations through frozen encoder → soft tokens
3. Decoder sees `[soft_tokens] + [suffix_embeds]` → predicts suffix tokens
4. Cross-entropy loss on suffix tokens

The FineWeb mixing prevents catastrophic forgetting of the pretraining
objective.

---

## Known Issues With This Approach

### 1. Self-Classification Is Circular

The subject model classifies its own input, then the decoder is trained to
recover that classification from the subject model's activations. This is
somewhat circular — the "correct" answers reflect the subject model's
surface-level text understanding, not its deeper internal beliefs or latent
representations.

The paper avoids this by using an external oracle (o3/Claude Sonnet) to
generate questions and GPT-4.1-mini to verify answer consistency.

### 2. Noisy Labels From Fuzzy Matching

The subject model's classification output is free-form text matched against a
fixed pool. When no pool entry matches, a random fallback is used. This
introduces label noise — the decoder is trained on some fraction of incorrect
answers.

### 3. Single-Token Answers Don't Teach Grounding

The answer is always one letter (`" A"`, `" B"`, etc.). With only 1
supervised token per example, the loss signal is weak. The decoder can learn
the marginal distribution of answers (e.g., "C" is most common for sentiment
because most web text is neutral) without actually reading the soft tokens.

The paper's SynthSys uses 60% open-ended questions with multi-token
descriptive answers, which forces the decoder to condition on the encoded
activations.

### 4. Text Properties ≠ Model Beliefs

Our questions ask about surface text properties (topic, sentiment, domain).
The paper's questions ask about model beliefs about user attributes (gender,
age, employment). The latter is more interesting for interpretability because
it probes what the model *infers* beyond what's literally in the text.

---

## What the Paper Does Differently (SynthSys)

For reference, the paper's QA generation pipeline:

1. **Prompt collection**: Dialogues where the subject model makes assumptions
   about user attributes (age, gender, diet, religion, etc.)

2. **Oracle QA generation**: o3 and Claude Sonnet (50/50) generate questions
   and answers about each of 80 user attributes

3. **Question format**: 60% open-ended, 25% multiple-choice, 15% yes/no

4. **Consistency verification**: GPT-4.1-mini checks that the subject model's
   behavior is consistent with the ground-truth attribute value. Only
   consistent examples are kept.

5. **Scale**: 78 attributes for training, 2 held out (gender, age) for
   evaluation. 78,964 verified QA pairs.

6. **Training**: Encoder frozen, decoder LoRA trained for 4,000 steps with
   50% FineWeb mixing. Same hyperparameters as pretraining.
