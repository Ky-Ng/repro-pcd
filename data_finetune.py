"""SynthSys-style QA data for PCD fine-tuning (Stage 2).

Generates user-attribute QA pairs following the SynthSys(8B) approach from
Choi et al. 2025 / Huang et al. 2025 (arXiv:2512.15712, Figure 6):

  1. Define user attributes with 4 values each (for A/B/C/D MCQ)
  2. For each (attribute, value, user_question) triple:
     - Format as: <SYSTEM> The user is {value} <USER> {question}
     - Run subject model to generate a response
     - Use Claude Haiku as LLM judge to check consistency
     - If consistent: keep the example
  3. Build MCQ for decoder: "What does the model think the {attr} of the user is?"
  4. During training, encoder sees only USER-position activations (not system)

The decoder learns to decode the subject model's latent user model from its
sparse-encoded activations into MCQ answers about user attributes.
"""

import os
import json
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import anthropic

from config import PCDConfig


# ---------------------------------------------------------------------------
# User attributes: 18 training + 2 held-out for evaluation
# Each has exactly 4 values for A/B/C/D multiple-choice
# ---------------------------------------------------------------------------

TRAIN_ATTRIBUTES = {
    "ethnicity": ["White", "Black", "Asian", "Hispanic"],
    "profession": ["Engineer", "Teacher", "Doctor", "Artist"],
    "dietary restriction": ["Vegan", "Vegetarian", "Omnivore", "Pescatarian"],
    "religion": ["Christian", "Muslim", "Atheist", "Buddhist"],
    "education level": ["High school", "Bachelor's degree", "Master's degree", "PhD"],
    "income level": ["Low income", "Middle income", "High income", "Very high income"],
    "marital status": ["Single", "Married", "Divorced", "Widowed"],
    "political leaning": ["Liberal", "Conservative", "Moderate", "Libertarian"],
    "primary hobby": ["Sports", "Reading", "Gaming", "Cooking"],
    "personality type": ["Introverted", "Extroverted", "Ambivert", "Reserved"],
    "fitness level": ["Sedentary", "Moderately active", "Very active", "Athletic"],
    "living situation": ["Urban apartment", "Suburban house", "Rural area", "College dorm"],
    "pet ownership": ["Dog owner", "Cat owner", "No pets", "Multiple pets"],
    "music preference": ["Pop", "Rock", "Classical", "Hip-hop"],
    "travel experience": ["Rarely travels", "Domestic traveler", "International traveler", "Digital nomad"],
    "tech savviness": ["Beginner", "Intermediate", "Advanced", "Expert"],
    "parental status": ["No children", "Young children", "Teenage children", "Adult children"],
    "sleep schedule": ["Early bird", "Night owl", "Irregular schedule", "Standard schedule"],
}

EVAL_ATTRIBUTES = {
    "gender": ["Female", "Male", "Non-binary", "Prefer not to say"],
    "age": ["18-25", "26-40", "41-60", "Over 60"],
}

# ---------------------------------------------------------------------------
# User questions per attribute — designed to elicit attribute-revealing
# responses from the subject model when primed with a system prompt
# ---------------------------------------------------------------------------

USER_QUESTIONS = {
    "ethnicity": [
        "What are some traditional dishes I should learn to cook?",
        "I'm looking for cultural events to attend this weekend, any ideas?",
        "What holidays should I prepare for this year?",
        "Can you recommend some music from my cultural background?",
        "What's a meaningful gift I could give to my family?",
    ],
    "profession": [
        "I'm stressed about work, any advice?",
        "What skills should I develop for career growth?",
        "How should I handle a difficult colleague?",
        "What's the best way to organize my workday?",
        "I need to give a presentation tomorrow, any tips?",
    ],
    "dietary restriction": [
        "What should I make for dinner tonight?",
        "Can you recommend a restaurant for a special occasion?",
        "What's a good protein source for my diet?",
        "I'm planning a dinner party, what should I serve?",
        "What snacks should I bring on a road trip?",
    ],
    "religion": [
        "I'm going through a difficult time, how can I find peace?",
        "What should I consider when planning my wedding ceremony?",
        "How do I teach my children about our values?",
        "What community activities should I participate in?",
        "How should I observe the upcoming holiday season?",
    ],
    "education level": [
        "I want to learn something new, what do you suggest?",
        "How should I approach a complex research problem?",
        "What books would you recommend for personal growth?",
        "I'm thinking about furthering my education, any advice?",
        "How do I stay current in my field?",
    ],
    "income level": [
        "I'm planning a vacation, what are my options?",
        "What's the best way to save for retirement?",
        "I need to buy a new car, what should I consider?",
        "How should I budget for the holidays?",
        "What kind of housing should I be looking for?",
    ],
    "marital status": [
        "I'm feeling lonely lately, any suggestions?",
        "How should I spend my weekends?",
        "What's important when making big life decisions?",
        "How do I balance my personal life with work?",
        "I'm thinking about my future, what should I plan for?",
    ],
    "political leaning": [
        "What do you think about the current state of education?",
        "How should communities address homelessness?",
        "What's the best approach to healthcare reform?",
        "How should we think about immigration policy?",
        "What's your view on government regulation of technology?",
    ],
    "primary hobby": [
        "What should I do this weekend for fun?",
        "I have some free time, how should I spend it?",
        "What equipment should I invest in for my hobby?",
        "How can I meet people who share my interests?",
        "What's a good way to improve at my favorite activity?",
    ],
    "personality type": [
        "I'm going to a party where I don't know many people, any tips?",
        "How should I recharge after a long week?",
        "What's the best work environment for me?",
        "How should I approach making new friends?",
        "What kind of vacation would suit me best?",
    ],
    "fitness level": [
        "What exercise routine would you recommend for me?",
        "How should I prepare for a physical challenge?",
        "What should I eat before and after working out?",
        "How can I stay motivated with my fitness goals?",
        "What sports or activities should I try?",
    ],
    "living situation": [
        "How should I decorate my living space?",
        "What's the best way to commute to work?",
        "How can I make my home more comfortable?",
        "What should I consider for my next move?",
        "How do I deal with noise from neighbors?",
    ],
    "pet ownership": [
        "I'm feeling stressed, how can my home life help?",
        "What should I plan for when going on vacation?",
        "How should I set up my daily routine?",
        "What's the best way to stay active at home?",
        "I'm thinking about making a lifestyle change, any advice?",
    ],
    "music preference": [
        "What concerts should I look for this summer?",
        "Can you recommend some new music for me?",
        "What's a good playlist for a road trip?",
        "How can I discover new artists I might like?",
        "What instrument should I learn to play?",
    ],
    "travel experience": [
        "Where should I go for my next trip?",
        "How do I plan an itinerary for a new destination?",
        "What travel tips would you give me?",
        "How should I pack for an upcoming trip?",
        "What kind of accommodation should I look for?",
    ],
    "tech savviness": [
        "My computer is running slowly, what should I do?",
        "How should I back up my important files?",
        "What's the best way to stay safe online?",
        "I need to set up a home network, any advice?",
        "What tools should I use for productivity?",
    ],
    "parental status": [
        "How should I plan my weekend?",
        "What's important when making financial decisions?",
        "How do I manage my time effectively?",
        "What should I prioritize in my daily routine?",
        "How can I maintain a healthy work-life balance?",
    ],
    "sleep schedule": [
        "When is the best time for me to exercise?",
        "How should I structure my work hours?",
        "What evening routine would you recommend?",
        "How can I be more productive during the day?",
        "What should I do when I can't fall asleep?",
    ],
}


def _format_chat_input(
    tokenizer,
    system_prompt: str,
    user_message: str,
) -> tuple[list[int], list[int]]:
    """Format system + user prompt using Qwen2.5 chat template.

    Returns separate token ID lists for system and user portions so we can
    extract activations only for the user tokens.

    Returns:
        (system_ids, user_ids) — token ID lists (no overlap)
    """
    # Build the full chat-formatted input
    messages_with_system = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    full_ids = tokenizer.apply_chat_template(
        messages_with_system, tokenize=True, add_generation_prompt=True,
        return_dict=False,
    )

    # Build system-only to find the split point
    messages_system_only = [
        {"role": "system", "content": system_prompt},
    ]
    system_ids = tokenizer.apply_chat_template(
        messages_system_only, tokenize=True, add_generation_prompt=False,
        return_dict=False,
    )

    # User portion = everything after system tokens
    user_ids = full_ids[len(system_ids):]

    return system_ids, user_ids


@torch.no_grad()
def _generate_response(
    subject_model,
    tokenizer,
    system_ids: list[int],
    user_ids: list[int],
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> str:
    """Generate subject model response given system + user prompt.

    Args:
        subject_model: SubjectModel wrapper (has .generate() and .model)
    """
    full_ids = system_ids + user_ids
    input_tensor = torch.tensor([full_ids], device=device)
    output = subject_model.generate(input_tensor, max_new_tokens=max_new_tokens)
    response_ids = output[0][len(full_ids):]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def _check_consistency(
    client: anthropic.Anthropic,
    attribute_name: str,
    attribute_value: str,
    user_question: str,
    response: str,
) -> bool:
    """Use Claude Haiku as LLM judge to check if the subject model's response
    is consistent with the given user attribute.

    This is the "noisy labeling" step from the paper (which uses GPT-4.1-mini).
    We use Haiku as a capable, fast, and cheap external judge.

    A response is "consistent" if it reflects, acknowledges, or is tailored
    to the attribute in any way — indicating the subject model's activations
    encode information about this user attribute.
    """
    judge_prompt = (
        f"A user described as having the attribute '{attribute_name}: {attribute_value}' "
        f"asked an AI assistant: \"{user_question}\"\n\n"
        f"The assistant responded: \"{response[:500]}\"\n\n"
        f"Is the assistant's response consistent with the assumption that the user "
        f"is '{attribute_value}'? A response is consistent if it reflects, acknowledges, "
        f"or is tailored to this attribute in any way — for example, recommending vegan "
        f"food for a vegan user, or suggesting activities appropriate for the user's "
        f"described characteristics.\n\n"
        f"Answer with just Yes or No."
    )

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        judge_response = msg.content[0].text.strip().lower()
        return judge_response.startswith("yes")
    except anthropic.RateLimitError:
        time.sleep(2)
        return _check_consistency(client, attribute_name, attribute_value, user_question, response)
    except Exception as e:
        print(f"  Judge error: {e}")
        return False


def _build_mcq(attribute_name: str, values: list[str], correct_value: str) -> tuple[str, str]:
    """Build a multiple-choice question and answer for the decoder.

    Returns:
        (question_text, answer_letter) e.g. ("What does the model think...", " A")
    """
    correct_idx = values.index(correct_value)
    correct_letter = chr(65 + correct_idx)  # A/B/C/D

    question = f"What does the model think the {attribute_name} of the user is?\n"
    for i, val in enumerate(values):
        question += f"({chr(65 + i)}) {val} "
    question += "\nAnswer:"

    return question, f" {correct_letter}"


class SynthSysDataset(Dataset):
    """Dataset of SynthSys-style QA pairs for fine-tuning."""

    def __init__(
        self,
        qa_pairs: list[dict],
        tokenizer,
        user_middle_len: int = 32,
        max_q_len: int = 80,
        max_a_len: int = 8,
    ):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.user_middle_len = user_middle_len
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        item = self.qa_pairs[idx]

        system_ids = item["system_ids"]
        user_ids = item["user_ids"]

        # Pad/truncate user_ids to user_middle_len
        if len(user_ids) > self.user_middle_len:
            user_ids = user_ids[:self.user_middle_len]
        elif len(user_ids) < self.user_middle_len:
            user_ids = user_ids + [self.tokenizer.pad_token_id] * (self.user_middle_len - len(user_ids))

        # Tokenize question
        q_ids = self.tokenizer.encode(item["question"], add_special_tokens=False)
        q_ids = q_ids[:self.max_q_len]
        q_ids = q_ids + [self.tokenizer.pad_token_id] * (self.max_q_len - len(q_ids))

        # Tokenize answer
        a_ids = self.tokenizer.encode(item["answer"], add_special_tokens=False)
        a_ids = a_ids[:self.max_a_len]
        a_ids = a_ids + [self.tokenizer.pad_token_id] * (self.max_a_len - len(a_ids))

        return {
            "system_ids": torch.tensor(system_ids, dtype=torch.long),
            "user_ids": torch.tensor(user_ids, dtype=torch.long),
            "system_len": len(item["system_ids"]),
            "user_len": min(len(item["user_ids"]), self.user_middle_len),
            "question_ids": torch.tensor(q_ids, dtype=torch.long),
            "answer_ids": torch.tensor(a_ids, dtype=torch.long),
        }


def _collate_synthsys(batch: list[dict]) -> dict:
    """Custom collate that pads system_ids to the max length in the batch.

    System prompts vary slightly in length across attributes, so we
    right-pad system_ids to the batch maximum. Since all system prompts
    are similar length in practice (same template), padding is minimal.
    """
    max_sys_len = max(item["system_ids"].shape[0] for item in batch)

    # Qwen2.5 eos_token_id = 151643 (used as pad token)
    PAD_ID = 151643

    collated = {
        "question_ids": torch.stack([item["question_ids"] for item in batch]),
        "answer_ids": torch.stack([item["answer_ids"] for item in batch]),
        "user_ids": torch.stack([item["user_ids"] for item in batch]),
        "system_len": torch.tensor([item["system_len"] for item in batch]),
        "user_len": torch.tensor([item["user_len"] for item in batch]),
    }

    # Right-pad system_ids so they all have the same length in the batch.
    # Right-padding keeps the system tokens left-aligned (position 0 onward),
    # which matches how the subject model expects them.
    padded_system = []
    for item in batch:
        sys = item["system_ids"]
        if sys.shape[0] < max_sys_len:
            padding = torch.full(
                (max_sys_len - sys.shape[0],),
                fill_value=PAD_ID,
                dtype=torch.long,
            )
            sys = torch.cat([sys, padding])
        padded_system.append(sys)
    collated["system_ids"] = torch.stack(padded_system)

    return collated


def prepare_qa_data(
    config: PCDConfig,
    subject_model,
    num_examples: int | None = None,
    cache_path: str | None = None,
    include_eval: bool = False,
) -> SynthSysDataset:
    """Generate SynthSys-style QA fine-tuning data.

    For each (attribute, value, question) triple:
      1. Format system+user prompt using chat template
      2. Generate subject model response
      3. Check consistency with LLM judge
      4. If consistent, build MCQ and add to dataset

    Args:
        config: PCD config
        subject_model: frozen subject model (for generation + judging)
        num_examples: target number of QA pairs (default from config)
        cache_path: path to cache generated QA pairs
        include_eval: if True, also generate from eval attributes (for testing)

    Returns:
        SynthSysDataset ready for DataLoader
    """
    if num_examples is None:
        num_examples = config.synthsys_num_examples
    if cache_path is None:
        cache_path = config.synthsys_cache_path

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check cache
    if os.path.exists(cache_path):
        print(f"Loading cached SynthSys QA data from {cache_path}")
        with open(cache_path) as f:
            qa_pairs = json.load(f)
        print(f"Loaded {len(qa_pairs)} QA pairs")
        return SynthSysDataset(qa_pairs, tokenizer, config.synthsys_user_middle_len)

    print(f"Generating SynthSys-style QA pairs (target: {num_examples})...")
    print("Using Claude Haiku as consistency judge...")

    # Initialize Haiku client for consistency judging
    haiku_client = anthropic.Anthropic()

    attributes = dict(TRAIN_ATTRIBUTES)
    if include_eval:
        attributes.update(EVAL_ATTRIBUTES)

    qa_pairs = []
    n_generated = 0
    n_consistent = 0
    n_inconsistent = 0

    # Build all (attribute, value, question) triples and shuffle
    triples = []
    for attr_name, values in attributes.items():
        questions = USER_QUESTIONS.get(attr_name, [])
        if not questions:
            continue
        for value in values:
            for question in questions:
                triples.append((attr_name, values, value, question))

    random.shuffle(triples)

    # We may need to cycle through triples multiple times to hit num_examples
    triple_idx = 0
    pbar = tqdm(total=num_examples, desc="Generating SynthSys QA pairs")

    while len(qa_pairs) < num_examples and triple_idx < len(triples) * 3:
        attr_name, values, value, question = triples[triple_idx % len(triples)]
        triple_idx += 1

        system_prompt = f"The user is {value}."

        # Tokenize system + user
        system_ids, user_ids = _format_chat_input(tokenizer, system_prompt, question)

        # Generate subject model response
        response = _generate_response(
            subject_model,
            tokenizer,
            system_ids,
            user_ids,
            max_new_tokens=config.synthsys_max_response_tokens,
            device=config.device,
        )
        n_generated += 1

        if len(response.strip()) < 10:
            continue

        # Consistency check via Haiku
        is_consistent = _check_consistency(
            haiku_client,
            attr_name,
            value,
            question,
            response,
        )

        if not is_consistent:
            n_inconsistent += 1
            continue

        n_consistent += 1

        # Build MCQ
        mcq_question, mcq_answer = _build_mcq(attr_name, values, value)

        qa_pairs.append({
            "system_ids": system_ids,
            "user_ids": user_ids,
            "question": mcq_question,
            "answer": mcq_answer,
            "attribute": attr_name,
            "value": value,
            "user_question": question,
            "response": response[:200],  # truncate for storage
        })
        pbar.update(1)

    pbar.close()

    consistency_rate = n_consistent / max(n_generated, 1) * 100
    print(f"\nGeneration complete:")
    print(f"  Total generated: {n_generated}")
    print(f"  Consistent (kept): {n_consistent} ({consistency_rate:.1f}%)")
    print(f"  Inconsistent (discarded): {n_inconsistent}")
    print(f"  Final dataset size: {len(qa_pairs)}")

    # Cache
    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(qa_pairs, f)
    print(f"Cached SynthSys QA pairs to {cache_path}")

    return SynthSysDataset(qa_pairs, tokenizer, config.synthsys_user_middle_len)


def get_qa_dataloader(dataset: SynthSysDataset, config: PCDConfig, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for SynthSys QA fine-tuning data."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,  # avoid pickle issues with tokenizer
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_synthsys,
    )
