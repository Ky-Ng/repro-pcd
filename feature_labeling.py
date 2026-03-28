"""Automated interpretability pipeline for PCD encoder concepts.

Collects top-activating examples per concept, labels them via Claude API,
and scores the labels for faithfulness.

Usage:
    # Step 1: Collect top-activating examples (local, no API key needed)
    python feature_labeling.py collect --checkpoint checkpoints/step_5000

    # Step 2: Label concepts via Claude API
    python feature_labeling.py label --examples concept_examples.json

    # Step 3: Score labels against held-out examples
    python feature_labeling.py score --labels concept_labels.json

    # All-in-one (requires API key)
    python feature_labeling.py run --checkpoint checkpoints/step_5000

    # Quick test on a few concepts
    python feature_labeling.py run --checkpoint checkpoints/step_5000 --max-concepts 10
"""

import argparse
import heapq
import json
import os
import time
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from config import PCDConfig
from model_subject import SubjectModel
from model_encoder import SparseEncoder
from data import prepare_data, get_dataloader

OUTPUT_DIR = "feature_annotations"


# ---------------------------------------------------------------------------
# Step 1: Collect top-activating examples
# ---------------------------------------------------------------------------

@dataclass(order=True)
class ActivatingExample:
    """A single example stored in a per-concept max-heap (min-heap by negated score)."""
    max_val: float  # used for heap ordering
    data: dict = field(compare=False)


def collect_top_examples(
    config: PCDConfig,
    encoder_path: str,
    top_k: int = 20,
    held_out_k: int = 25,
    max_batches: int | None = None,
) -> dict:
    """Run corpus through subject model + encoder, collect top examples per concept.

    Args:
        config: PCD configuration
        encoder_path: Path to encoder checkpoint
        top_k: Number of top-activating examples to keep per concept
        held_out_k: Number of held-out activating examples for scoring
        max_batches: Limit number of batches (for testing)

    Returns:
        dict with "concepts" (top examples) and "held_out" (scoring examples)
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading subject model...")
    subject = SubjectModel(config)

    print("Loading encoder...")
    encoder = SparseEncoder(config).to(config.device).to(config.dtype)
    encoder.load_state_dict(
        torch.load(encoder_path, map_location=config.device, weights_only=True)
    )
    encoder.eval()

    print("Loading data...")
    dataset = prepare_data(config)
    dataloader = get_dataloader(dataset, config, shuffle=False)

    n_concepts = config.num_concepts
    total_k = top_k + held_out_k  # collect extra for held-out split

    # Min-heaps per concept (store negative max_val for min-heap behavior)
    heaps: list[list] = [[] for _ in range(n_concepts)]
    # Track how often each concept fires across the corpus
    fire_counts = torch.zeros(n_concepts, dtype=torch.long)

    print(f"Collecting top-{total_k} examples for {n_concepts} concepts...")
    n_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=n_batches, desc="Collecting")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            prefix_ids = batch["prefix_ids"].to(config.device)
            middle_ids = batch["middle_ids"].to(config.device)

            # Get subject model activations
            subject_input = torch.cat([prefix_ids, middle_ids], dim=1)
            activations = subject.get_middle_activations(
                subject_input, config.prefix_len, config.middle_len
            )

            # Normalize and project to concept space
            normed = encoder._normalize_activations(activations)
            pre_act = encoder.W_enc(normed)  # [B, T, n_concepts]

            B, T, M = pre_act.shape

            # For each example in batch, find which concepts fire (enter top-k)
            top_vals_batch, top_idx_batch = torch.topk(pre_act, config.topk, dim=-1)  # [B, T, k]

            # Update fire counts
            for b in range(B):
                fired = top_idx_batch[b].unique().cpu()
                fire_counts[fired] += 1

            # Pre-compute per-concept max activation and position
            max_per_concept = pre_act.max(dim=1)  # values: [B, M], indices: [B, M]
            max_vals = max_per_concept.values  # [B, M]
            max_positions = max_per_concept.indices  # [B, M]

            # Move to CPU once for the whole batch
            pre_act_cpu = pre_act.cpu()
            top_idx_cpu = top_idx_batch.reshape(B, -1).cpu()  # [B, T*k]

            for b in range(B):
                # Decode the full window for this example
                full_ids = torch.cat([prefix_ids[b], middle_ids[b]]).cpu().tolist()
                middle_ids_list = middle_ids[b].cpu().tolist()
                text = tokenizer.decode(full_ids, skip_special_tokens=False)
                middle_text = tokenizer.decode(middle_ids_list, skip_special_tokens=False)
                tokens = [tokenizer.decode([tid]) for tid in middle_ids_list]

                # Only iterate concepts that actually fired (entered top-k)
                # at any position for this example — at most T*k = 256
                fired_concepts = top_idx_cpu[b].unique().tolist()

                for c in fired_concepts:
                    val = max_vals[b, c].item()

                    # Check if this example is worth storing
                    heap = heaps[c]
                    if len(heap) >= total_k and val <= heap[0].max_val:
                        continue

                    # Per-token activations for this concept across middle positions
                    per_token_acts = pre_act_cpu[b, :, c].tolist()

                    example_data = {
                        "text": text,
                        "middle_text": middle_text,
                        "tokens": tokens,
                        "activations": per_token_acts,
                        "max_activation": val,
                        "max_token_idx": max_positions[b, c].item(),
                        "batch_idx": batch_idx,
                        "example_idx": b,
                    }

                    if len(heap) < total_k:
                        heapq.heappush(heap, ActivatingExample(val, example_data))
                    else:
                        heapq.heapreplace(heap, ActivatingExample(val, example_data))

    # Split into top examples (for labeling) and held-out (for scoring)
    concepts = {}
    held_out = {}
    for c in range(n_concepts):
        examples = sorted(heaps[c], key=lambda x: -x.max_val)
        all_data = [e.data for e in examples]
        concepts[c] = {
            "concept_id": c,
            "n_fires": fire_counts[c].item(),
            "examples": all_data[:top_k],
        }
        held_out[c] = {
            "concept_id": c,
            "activating": all_data[top_k:top_k + held_out_k],
        }

    return {"concepts": concepts, "held_out": held_out, "fire_counts": fire_counts.tolist()}


# ---------------------------------------------------------------------------
# Step 2: Label concepts via Claude API
# ---------------------------------------------------------------------------

def format_example_for_prompt(example: dict, show_context: bool = True) -> str:
    """Format a single example with per-token activations for the LLM prompt."""
    tokens = example["tokens"]
    acts = example["activations"]
    max_idx = example["max_token_idx"]

    parts = []
    for i, (tok, act) in enumerate(zip(tokens, acts)):
        tok_clean = tok.replace("\n", "\\n")
        if i == max_idx:
            parts.append(f">>>[{act:.3f}] {tok_clean}<<<")
        else:
            parts.append(f"[{act:.3f}] {tok_clean}")

    formatted = "  " + "  ".join(parts)

    if show_context and example.get("text"):
        return f"  Context: {example['text'][:120]}...\n{formatted}"
    return formatted


LABEL_PROMPT_TEMPLATE = """You are analyzing features in a sparse encoder trained on a language model's internal activations. Each feature corresponds to a concept the model has learned to represent.

Below are text passages where Feature {concept_id} activates strongly. Each token is annotated with the feature's activation value in brackets. Higher values mean stronger activation. The token with peak activation is marked with >>>arrows<<<.

This feature fired {n_fires} times across the corpus.

{examples_text}

Based on these examples, describe what this feature detects.

Respond in EXACTLY this format (no other text):
LABEL: <5-10 word label>
DESCRIPTION: <1-2 paragraph description of what activates this feature, including any nuances about when it fires strongly vs weakly>
CONFIDENCE: <high or medium or low>"""


def build_label_prompt(concept_data: dict) -> str:
    """Build the labeling prompt for a single concept."""
    examples = concept_data["examples"]
    n_fires = concept_data.get("n_fires", "unknown")

    examples_text = ""
    for i, ex in enumerate(examples[:20]):
        examples_text += f"\nExample {i + 1} (max activation: {ex['max_activation']:.3f}):\n"
        examples_text += format_example_for_prompt(ex) + "\n"

    return LABEL_PROMPT_TEMPLATE.format(
        concept_id=concept_data["concept_id"],
        n_fires=n_fires,
        examples_text=examples_text,
    )


def parse_label_response(response_text: str) -> dict:
    """Parse the structured label response from Claude."""
    result = {"label": "", "description": "", "confidence": "low", "raw_response": response_text}

    for line in response_text.split("\n"):
        line = line.strip()
        if line.startswith("LABEL:"):
            result["label"] = line[len("LABEL:"):].strip()
        elif line.startswith("DESCRIPTION:"):
            result["description"] = line[len("DESCRIPTION:"):].strip()
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line[len("CONFIDENCE:"):].strip().lower()

    # Handle multi-line description: everything between DESCRIPTION: and CONFIDENCE:
    if "DESCRIPTION:" in response_text and "CONFIDENCE:" in response_text:
        desc_start = response_text.index("DESCRIPTION:") + len("DESCRIPTION:")
        desc_end = response_text.index("CONFIDENCE:")
        result["description"] = response_text[desc_start:desc_end].strip()

    return result


def label_concepts(
    concepts: dict,
    model: str = "claude-sonnet-4-6",
    max_concepts: int | None = None,
    concurrency: int = 10,
) -> dict:
    """Label concepts using the Claude API.

    Args:
        concepts: dict of concept_id -> concept data with examples
        model: Claude model to use
        max_concepts: Limit number of concepts to label (for testing)
        concurrency: Number of concurrent API calls (for future async impl)

    Returns:
        dict of concept_id -> label data
    """
    import anthropic
    client = anthropic.Anthropic()

    concept_ids = sorted(concepts.keys(), key=int)
    if max_concepts is not None:
        concept_ids = concept_ids[:max_concepts]

    labels = {}
    for concept_id in tqdm(concept_ids, desc="Labeling concepts"):
        concept_data = concepts[concept_id]

        # Skip concepts with no examples
        if not concept_data["examples"]:
            labels[concept_id] = {
                "concept_id": concept_id,
                "label": "dead concept (never fired)",
                "description": "This concept never activated on any passage in the corpus.",
                "confidence": "high",
                "detection_score": None,
            }
            continue

        prompt = build_label_prompt(concept_data)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text
            parsed = parse_label_response(response_text)
            parsed["concept_id"] = concept_id
            labels[concept_id] = parsed

        except Exception as e:
            print(f"  Error labeling concept {concept_id}: {e}")
            labels[concept_id] = {
                "concept_id": concept_id,
                "label": f"ERROR: {str(e)[:100]}",
                "description": "",
                "confidence": "low",
                "error": str(e),
            }

        # Rate limiting
        time.sleep(0.1)

    return labels


# ---------------------------------------------------------------------------
# Step 3: Detection scoring
# ---------------------------------------------------------------------------

SCORE_PROMPT_TEMPLATE = """A feature in a neural network is described as:
"{label}"

Full description: "{description}"

Given this text passage:
"{passage_text}"

On a scale of 0 to 10, how strongly would you expect this feature to activate on this passage? Respond with ONLY a single number (0-10), nothing else."""


def score_labels(
    labels: dict,
    held_out: dict,
    concepts: dict,
    model: str = "claude-sonnet-4-6",
    n_negative: int = 25,
    max_concepts: int | None = None,
) -> dict:
    """Score concept labels against held-out examples.

    For each concept:
    - Show the label + held-out activating passages -> expect high predicted score
    - Show the label + random non-activating passages -> expect low predicted score
    - Compute correlation between predicted and actual

    Args:
        labels: dict of concept_id -> label data
        held_out: dict of concept_id -> held-out activating examples
        concepts: dict of concept_id -> concept data (for sampling negatives)
        model: Claude model to use
        n_negative: Number of non-activating passages to sample per concept
        max_concepts: Limit number of concepts to score

    Returns:
        dict of concept_id -> scored label data
    """
    import anthropic
    import random
    client = anthropic.Anthropic()

    # Collect all passages for negative sampling
    all_passages = []
    for cid, cdata in concepts.items():
        for ex in cdata["examples"]:
            all_passages.append(ex["middle_text"])

    concept_ids = sorted(labels.keys(), key=int)
    if max_concepts is not None:
        concept_ids = concept_ids[:max_concepts]

    scored = {}
    for concept_id in tqdm(concept_ids, desc="Scoring labels"):
        label_data = labels[concept_id]

        if not label_data.get("label") or label_data["label"].startswith("dead concept"):
            label_data["detection_score"] = None
            scored[concept_id] = label_data
            continue

        label = label_data["label"]
        description = label_data.get("description", "")

        # Positive examples (held-out activating passages)
        positives = held_out.get(concept_id, {}).get("activating", [])
        # Negative examples (random passages from other concepts)
        negatives_text = random.sample(all_passages, min(n_negative, len(all_passages)))

        predictions = []
        actuals = []

        # Score positive examples
        for ex in positives[:25]:
            passage = ex["middle_text"]
            prompt = SCORE_PROMPT_TEMPLATE.format(
                label=label, description=description, passage_text=passage[:300],
            )
            try:
                response = client.messages.create(
                    model=model, max_tokens=8,
                    messages=[{"role": "user", "content": prompt}],
                )
                pred = float(response.content[0].text.strip())
                predictions.append(pred)
                actuals.append(1.0)  # positive
            except (ValueError, Exception):
                pass
            time.sleep(0.05)

        # Score negative examples
        for passage in negatives_text[:25]:
            prompt = SCORE_PROMPT_TEMPLATE.format(
                label=label, description=description, passage_text=passage[:300],
            )
            try:
                response = client.messages.create(
                    model=model, max_tokens=8,
                    messages=[{"role": "user", "content": prompt}],
                )
                pred = float(response.content[0].text.strip())
                predictions.append(pred)
                actuals.append(0.0)  # negative
            except (ValueError, Exception):
                pass
            time.sleep(0.05)

        # Compute correlation
        if len(predictions) >= 4:
            import numpy as np
            preds = np.array(predictions)
            acts = np.array(actuals)
            if preds.std() > 0 and acts.std() > 0:
                corr = np.corrcoef(preds, acts)[0, 1]
            else:
                corr = 0.0
            label_data["detection_score"] = round(float(corr), 4)
            label_data["n_scored"] = len(predictions)
        else:
            label_data["detection_score"] = None
            label_data["n_scored"] = len(predictions)

        scored[concept_id] = label_data

    return scored


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------

def save_json(data: dict, path: str):
    """Save dict to JSON, converting int keys to strings."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    # JSON requires string keys
    serializable = {}
    for k, v in data.items():
        serializable[str(k)] = v
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Saved to {path}")


def load_json(path: str) -> dict:
    """Load JSON, converting string keys back to ints where possible."""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for k, v in data.items():
        try:
            result[int(k)] = v
        except ValueError:
            result[k] = v
    return result


def run_collect(args):
    """Run the collection phase."""
    config = PCDConfig()
    encoder_path = os.path.join(args.checkpoint, "encoder.pt")

    result = collect_top_examples(
        config,
        encoder_path,
        top_k=args.top_k,
        held_out_k=args.held_out_k,
        max_batches=args.max_batches,
    )

    save_json(result["concepts"], args.output_examples)
    save_json(result["held_out"], args.output_held_out)
    save_json({"fire_counts": result["fire_counts"]}, args.output_stats)

    # Summary
    n_with_examples = sum(1 for c in result["concepts"].values() if c["examples"])
    n_dead = sum(1 for c in result["fire_counts"] if c == 0)
    print(f"\nCollection complete:")
    print(f"  Concepts with examples: {n_with_examples}")
    print(f"  Dead concepts (never fired): {n_dead}")


def run_label(args):
    """Run the labeling phase."""
    concepts = load_json(args.examples)

    # When --max-concepts is set, prioritize the most active concepts
    if args.max_concepts is not None:
        sorted_by_fires = sorted(concepts.items(), key=lambda x: -x[1].get("n_fires", 0))
        concepts = dict(sorted_by_fires[:args.max_concepts])
        print(f"Selected top {len(concepts)} concepts by fire count")

    labels = label_concepts(
        concepts,
        model=args.model,
    )
    save_json(labels, args.output)

    # Summary
    n_labeled = sum(1 for l in labels.values() if l.get("label") and not l["label"].startswith("dead"))
    confidence_counts = {}
    for l in labels.values():
        c = l.get("confidence", "unknown")
        confidence_counts[c] = confidence_counts.get(c, 0) + 1
    print(f"\nLabeling complete:")
    print(f"  Labeled: {n_labeled}")
    print(f"  Confidence distribution: {confidence_counts}")


def run_score(args):
    """Run the scoring phase."""
    labels = load_json(args.labels)
    held_out = load_json(args.held_out)
    concepts = load_json(args.examples)

    scored = score_labels(
        labels,
        held_out,
        concepts,
        model=args.model,
        max_concepts=args.max_concepts,
    )
    save_json(scored, args.output)

    # Summary
    scores = [l["detection_score"] for l in scored.values() if l.get("detection_score") is not None]
    if scores:
        import numpy as np
        scores = np.array(scores)
        print(f"\nScoring complete:")
        print(f"  Mean detection score: {scores.mean():.3f}")
        print(f"  Excellent (>0.7): {(scores > 0.7).sum()}")
        print(f"  Good (0.5-0.7): {((scores > 0.5) & (scores <= 0.7)).sum()}")
        print(f"  Partial (0.3-0.5): {((scores > 0.3) & (scores <= 0.5)).sum()}")
        print(f"  Poor (<0.3): {(scores <= 0.3).sum()}")


def run_all(args):
    """Run the full pipeline: collect -> label -> score."""
    config = PCDConfig()
    encoder_path = os.path.join(args.checkpoint, "encoder.pt")

    # Step 1: Collect
    print("=" * 60)
    print("Step 1: Collecting top-activating examples")
    print("=" * 60)
    result = collect_top_examples(
        config,
        encoder_path,
        top_k=args.top_k,
        held_out_k=args.held_out_k,
        max_batches=args.max_batches,
    )

    examples_path = f"{OUTPUT_DIR}/concept_examples.json"
    held_out_path = f"{OUTPUT_DIR}/concept_held_out.json"
    save_json(result["concepts"], examples_path)
    save_json(result["held_out"], held_out_path)

    # Filter to concepts that actually fired
    active_concepts = {
        k: v for k, v in result["concepts"].items() if v["examples"]
    }
    if args.max_concepts:
        # Pick the most active concepts
        sorted_by_fires = sorted(active_concepts.items(), key=lambda x: -x[1]["n_fires"])
        active_concepts = dict(sorted_by_fires[:args.max_concepts])

    n_to_label = len(active_concepts)
    print(f"\n{n_to_label} concepts to label")

    # Step 2: Label
    print("\n" + "=" * 60)
    print("Step 2: Labeling concepts via Claude API")
    print("=" * 60)
    labels = label_concepts(
        active_concepts,
        model=args.model,
        max_concepts=args.max_concepts,
    )

    labels_path = f"{OUTPUT_DIR}/concept_labels.json"
    save_json(labels, labels_path)

    # Step 3: Score
    print("\n" + "=" * 60)
    print("Step 3: Scoring labels against held-out examples")
    print("=" * 60)
    scored = score_labels(
        labels,
        result["held_out"],
        result["concepts"],
        model=args.model,
        max_concepts=args.max_concepts,
    )

    final_path = f"{OUTPUT_DIR}/concept_labels_scored.json"
    save_json(scored, final_path)

    # Final summary
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"  Examples: {examples_path}")
    print(f"  Labels: {labels_path}")
    print(f"  Scored labels: {final_path}")

    # Print a few example labels
    print("\nSample labels:")
    for cid, ldata in list(scored.items())[:5]:
        score_str = f"{ldata['detection_score']:.2f}" if ldata.get('detection_score') is not None else "N/A"
        print(f"  Concept {cid}: {ldata.get('label', 'N/A')} (score: {score_str})")


def main():
    parser = argparse.ArgumentParser(description="PCD Feature Labeling Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Collect subcommand
    p_collect = subparsers.add_parser("collect", help="Collect top-activating examples")
    p_collect.add_argument("--checkpoint", type=str, default="checkpoints/step_5000")
    p_collect.add_argument("--top-k", type=int, default=20, help="Top examples per concept")
    p_collect.add_argument("--held-out-k", type=int, default=25, help="Held-out examples for scoring")
    p_collect.add_argument("--max-batches", type=int, default=None, help="Limit batches (for testing)")
    p_collect.add_argument("--output-examples", type=str, default=f"{OUTPUT_DIR}/concept_examples.json")
    p_collect.add_argument("--output-held-out", type=str, default=f"{OUTPUT_DIR}/concept_held_out.json")
    p_collect.add_argument("--output-stats", type=str, default=f"{OUTPUT_DIR}/concept_stats.json")
    p_collect.set_defaults(func=run_collect)

    # Label subcommand
    p_label = subparsers.add_parser("label", help="Label concepts via Claude API")
    p_label.add_argument("--examples", type=str, default=f"{OUTPUT_DIR}/concept_examples.json")
    p_label.add_argument("--model", type=str, default="claude-sonnet-4-6")
    p_label.add_argument("--max-concepts", type=int, default=None)
    p_label.add_argument("--output", type=str, default=f"{OUTPUT_DIR}/concept_labels.json")
    p_label.set_defaults(func=run_label)

    # Score subcommand
    p_score = subparsers.add_parser("score", help="Score labels via Claude API")
    p_score.add_argument("--labels", type=str, default=f"{OUTPUT_DIR}/concept_labels.json")
    p_score.add_argument("--held-out", type=str, default=f"{OUTPUT_DIR}/concept_held_out.json")
    p_score.add_argument("--examples", type=str, default=f"{OUTPUT_DIR}/concept_examples.json")
    p_score.add_argument("--model", type=str, default="claude-sonnet-4-6")
    p_score.add_argument("--max-concepts", type=int, default=None)
    p_score.add_argument("--output", type=str, default=f"{OUTPUT_DIR}/concept_labels_scored.json")
    p_score.set_defaults(func=run_score)

    # Run all subcommand
    p_run = subparsers.add_parser("run", help="Run full pipeline")
    p_run.add_argument("--checkpoint", type=str, default="checkpoints/step_5000")
    p_run.add_argument("--top-k", type=int, default=20)
    p_run.add_argument("--held-out-k", type=int, default=25)
    p_run.add_argument("--max-batches", type=int, default=None)
    p_run.add_argument("--max-concepts", type=int, default=None, help="Limit concepts (for testing)")
    p_run.add_argument("--model", type=str, default="claude-sonnet-4-6")
    p_run.set_defaults(func=run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
