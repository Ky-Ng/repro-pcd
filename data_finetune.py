"""Data loading for PCD fine-tuning (Stage 2).

The paper fine-tunes on SynthSys(8B) QA pairs about user attributes, mixing
in 50% FineWeb to prevent forgetting. Since SynthSys is not publicly available,
we construct QA pairs from FineWeb text by asking the subject model to generate
questions about text passages, then formatting them as multiple-choice QA.

Each QA example has:
  - input_text: the passage the subject model reads (prefix + middle tokens)
  - question: a natural language question about the passage content
  - answer: the correct answer string

During training the decoder sees:
  [soft_tokens_from_encoder] + [question_embeds] → predict [answer_tokens]
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
from tqdm import tqdm

from config import PCDConfig


# ---------------------------------------------------------------------------
# QA templates — attribute-style questions that probe what information the
# encoder captured from the subject model's activations, similar to SynthSys.
# Each template produces a question + multiple-choice answers.
# ---------------------------------------------------------------------------

ATTRIBUTE_TEMPLATES = [
    {
        "question": "What is the primary topic of the text?\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:",
        "field": "topic",
    },
    {
        "question": "What is the tone or sentiment of the text?\nA. Positive\nB. Negative\nC. Neutral\nD. Mixed\nAnswer:",
        "field": "sentiment",
    },
    {
        "question": "What domain does the text belong to?\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:",
        "field": "domain",
    },
    {
        "question": "Is the text formal or informal?\nA. Formal\nB. Informal\nC. Semi-formal\nD. Cannot determine\nAnswer:",
        "field": "formality",
    },
    {
        "question": "What type of content is this?\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:",
        "field": "content_type",
    },
    {
        "question": "What is the text primarily about?\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:",
        "field": "about",
    },
    {
        "question": "Does the text contain factual claims or opinions?\nA. Mostly factual\nB. Mostly opinions\nC. Mix of both\nD. Neither\nAnswer:",
        "field": "factuality",
    },
    {
        "question": "What is the intended audience of the text?\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:",
        "field": "audience",
    },
]

# Distractor pools for filling in multiple-choice options
TOPIC_POOL = [
    "science", "technology", "politics", "sports", "entertainment",
    "health", "education", "business", "travel", "food", "history",
    "nature", "art", "music", "literature", "philosophy", "religion",
    "law", "finance", "environment", "culture", "society",
]

DOMAIN_POOL = [
    "news", "academic", "blog", "social media", "encyclopedia",
    "technical documentation", "creative writing", "legal",
    "medical", "financial", "educational", "marketing",
]

CONTENT_TYPE_POOL = [
    "narrative", "expository", "argumentative", "descriptive",
    "instructional", "conversational", "analytical", "review",
    "report", "opinion piece", "tutorial", "reference material",
]

AUDIENCE_POOL = [
    "general public", "professionals", "students", "researchers",
    "children", "enthusiasts", "experts", "beginners",
]


def _generate_qa_from_text(
    text: str,
    subject_model,
    tokenizer,
    config: PCDConfig,
) -> dict | None:
    """Generate a QA pair from a text passage using the subject model.

    Instead of calling an oracle LLM (as the paper does with an 8B model),
    we use the subject model itself to classify text attributes, then
    construct multiple-choice questions about those attributes.

    Args:
        text: decoded text from the FineWeb passage
        subject_model: the frozen subject model
        tokenizer: shared tokenizer
        config: PCD config

    Returns:
        dict with 'prefix_ids', 'middle_ids', 'question', 'answer' or None
    """
    # Tokenize the passage
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < config.prefix_len + config.middle_len:
        return None

    prefix_ids = token_ids[:config.prefix_len]
    middle_ids = token_ids[config.prefix_len:config.prefix_len + config.middle_len]

    # Pick a random template
    template = random.choice(ATTRIBUTE_TEMPLATES)
    field = template["field"]

    # Use the subject model to generate an answer for the passage
    # We construct a classification prompt and take the first generated token
    passage_text = tokenizer.decode(prefix_ids + middle_ids, skip_special_tokens=True)

    if field == "topic":
        correct = _classify_topic(passage_text, subject_model, tokenizer, config)
        distractors = random.sample([t for t in TOPIC_POOL if t != correct], 3)
        options = [correct] + distractors
        random.shuffle(options)
        correct_letter = chr(65 + options.index(correct))  # A/B/C/D
        question = template["question"].format(
            a=options[0], b=options[1], c=options[2], d=options[3]
        )
        answer = f" {correct_letter}"
    elif field == "domain":
        correct = _classify_from_pool(passage_text, DOMAIN_POOL, subject_model, tokenizer, config)
        distractors = random.sample([t for t in DOMAIN_POOL if t != correct], 3)
        options = [correct] + distractors
        random.shuffle(options)
        correct_letter = chr(65 + options.index(correct))
        question = template["question"].format(
            a=options[0], b=options[1], c=options[2], d=options[3]
        )
        answer = f" {correct_letter}"
    elif field == "content_type":
        correct = _classify_from_pool(passage_text, CONTENT_TYPE_POOL, subject_model, tokenizer, config)
        distractors = random.sample([t for t in CONTENT_TYPE_POOL if t != correct], 3)
        options = [correct] + distractors
        random.shuffle(options)
        correct_letter = chr(65 + options.index(correct))
        question = template["question"].format(
            a=options[0], b=options[1], c=options[2], d=options[3]
        )
        answer = f" {correct_letter}"
    elif field == "audience":
        correct = _classify_from_pool(passage_text, AUDIENCE_POOL, subject_model, tokenizer, config)
        distractors = random.sample([t for t in AUDIENCE_POOL if t != correct], 3)
        options = [correct] + distractors
        random.shuffle(options)
        correct_letter = chr(65 + options.index(correct))
        question = template["question"].format(
            a=options[0], b=options[1], c=options[2], d=options[3]
        )
        answer = f" {correct_letter}"
    elif field == "about":
        # Free-form: generate a short summary as the correct answer,
        # and pick distractors from other passages
        correct = _generate_short_summary(passage_text, subject_model, tokenizer, config)
        distractors = random.sample(TOPIC_POOL, 3)
        options = [correct] + distractors
        random.shuffle(options)
        correct_letter = chr(65 + options.index(correct))
        question = template["question"].format(
            a=options[0], b=options[1], c=options[2], d=options[3]
        )
        answer = f" {correct_letter}"
    else:
        # Sentiment, formality, factuality — fixed options already in template
        question = template["question"]
        answer = _classify_fixed_options(passage_text, question, subject_model, tokenizer, config)

    return {
        "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
        "middle_ids": torch.tensor(middle_ids, dtype=torch.long),
        "question": question,
        "answer": answer,
    }


@torch.no_grad()
def _classify_topic(text, subject_model, tokenizer, config):
    """Use subject model to classify the topic of a passage."""
    prompt = f'Classify the topic of this text in one word: "{text[:200]}"\nTopic:'
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(config.device)
    output = subject_model.generate(input_ids, max_new_tokens=5)
    generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    # Map to nearest pool entry
    for topic in TOPIC_POOL:
        if topic in generated:
            return topic
    return random.choice(TOPIC_POOL)


@torch.no_grad()
def _classify_from_pool(text, pool, subject_model, tokenizer, config):
    """Classify text into one of the pool categories using the subject model."""
    options_str = ", ".join(pool)
    prompt = f'Classify this text into one of: [{options_str}]\nText: "{text[:200]}"\nCategory:'
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(config.device)
    output = subject_model.generate(input_ids, max_new_tokens=5)
    generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    for item in pool:
        if item in generated:
            return item
    return random.choice(pool)


@torch.no_grad()
def _generate_short_summary(text, subject_model, tokenizer, config):
    """Generate a short topic phrase from the subject model."""
    prompt = f'What is this text about in 2-3 words? "{text[:200]}"\nAbout:'
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(config.device)
    output = subject_model.generate(input_ids, max_new_tokens=8)
    generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    # Truncate to first phrase
    generated = generated.split("\n")[0].split(".")[0].strip()
    if len(generated) < 2:
        return random.choice(TOPIC_POOL)
    return generated[:40]


@torch.no_grad()
def _classify_fixed_options(text, question, subject_model, tokenizer, config):
    """For fixed-option questions, pick the answer using the subject model."""
    prompt = f'Text: "{text[:200]}"\n{question}'
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(config.device)
    output = subject_model.generate(input_ids, max_new_tokens=3)
    generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    # Extract letter answer
    for letter in ["A", "B", "C", "D"]:
        if letter in generated.upper()[:3]:
            return f" {letter}"
    return " A"


class QADataset(Dataset):
    """Dataset of pre-generated QA pairs for fine-tuning."""

    def __init__(self, qa_pairs: list[dict], tokenizer, max_q_len: int = 80, max_a_len: int = 8):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        item = self.qa_pairs[idx]

        # Tokenize question
        q_ids = self.tokenizer.encode(item["question"], add_special_tokens=False)
        q_ids = q_ids[:self.max_q_len]
        # Pad to max_q_len
        q_ids = q_ids + [self.tokenizer.pad_token_id] * (self.max_q_len - len(q_ids))

        # Tokenize answer
        a_ids = self.tokenizer.encode(item["answer"], add_special_tokens=False)
        a_ids = a_ids[:self.max_a_len]
        # Pad to max_a_len
        a_ids = a_ids + [self.tokenizer.pad_token_id] * (self.max_a_len - len(a_ids))

        return {
            "prefix_ids": item["prefix_ids"],
            "middle_ids": item["middle_ids"],
            "question_ids": torch.tensor(q_ids, dtype=torch.long),
            "answer_ids": torch.tensor(a_ids, dtype=torch.long),
            # Mask for question length (for proper attention masking)
            "question_len": len(self.tokenizer.encode(item["question"], add_special_tokens=False)),
        }


def prepare_qa_data(
    config: PCDConfig,
    subject_model,
    num_examples: int = 10_000,
    cache_path: str | None = None,
) -> QADataset:
    """Generate QA fine-tuning data from FineWeb passages.

    Uses the subject model to classify text attributes, creating
    SynthSys-style multiple-choice QA pairs.

    Args:
        config: PCD config
        subject_model: frozen subject model (for generating answers)
        num_examples: number of QA pairs to generate
        cache_path: path to cache generated QA pairs

    Returns:
        QADataset ready for DataLoader
    """
    if cache_path is None:
        cache_path = os.path.join(config.data_cache_dir, "qa_pairs.json")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check cache
    if os.path.exists(cache_path):
        print(f"Loading cached QA data from {cache_path}")
        with open(cache_path) as f:
            raw_pairs = json.load(f)
        # Reconstruct tensors
        qa_pairs = []
        for item in raw_pairs:
            qa_pairs.append({
                "prefix_ids": torch.tensor(item["prefix_ids"], dtype=torch.long),
                "middle_ids": torch.tensor(item["middle_ids"], dtype=torch.long),
                "question": item["question"],
                "answer": item["answer"],
            })
        print(f"Loaded {len(qa_pairs)} QA pairs")
        return QADataset(qa_pairs, tokenizer)

    print(f"Generating {num_examples} QA pairs from FineWeb...")

    # Load FineWeb windows (reuse from pretraining cache)
    fineweb_cache = os.path.join(config.data_cache_dir, "fineweb_windows.pt")
    if os.path.exists(fineweb_cache):
        windows = torch.load(fineweb_cache, weights_only=True)
    else:
        from data import prepare_data
        dataset = prepare_data(config, num_examples=num_examples * 2)
        windows = dataset.windows

    random.shuffle(windows)

    qa_pairs = []
    for window in tqdm(windows[:num_examples * 2], desc="Generating QA pairs"):
        if len(qa_pairs) >= num_examples:
            break

        # Decode the window text for classification
        text = tokenizer.decode(window.tolist(), skip_special_tokens=True)
        if len(text.strip()) < 30:
            continue

        pair = _generate_qa_from_text(text, subject_model, tokenizer, config)
        if pair is not None:
            qa_pairs.append(pair)

    print(f"Generated {len(qa_pairs)} QA pairs")

    # Cache (convert tensors to lists for JSON)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    serializable = []
    for item in qa_pairs:
        serializable.append({
            "prefix_ids": item["prefix_ids"].tolist(),
            "middle_ids": item["middle_ids"].tolist(),
            "question": item["question"],
            "answer": item["answer"],
        })
    with open(cache_path, "w") as f:
        json.dump(serializable, f)
    print(f"Cached QA pairs to {cache_path}")

    return QADataset(qa_pairs, tokenizer)


def get_qa_dataloader(dataset: QADataset, config: PCDConfig, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for QA fine-tuning data."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,  # avoid pickle issues with tokenizer
        pin_memory=True,
        drop_last=True,
    )
