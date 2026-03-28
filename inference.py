"""End-to-end PCD inference pipeline."""

import json
import os

import torch
from transformers import AutoTokenizer
from peft import PeftModel

from config import PCDConfig
from model_subject import SubjectModel
from model_encoder import SparseEncoder
from model_decoder import PCDDecoder


class PCDPipeline:
    """Run the full PCD pipeline: subject → encoder → decoder."""

    def __init__(
        self,
        config: PCDConfig,
        encoder_path: str,
        decoder_lora_path: str,
        feature_labels_path: str | None = "feature_annotations/concept_labels.json",
    ):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load subject model
        print("Loading subject model...")
        self.subject = SubjectModel(config)

        # Load encoder
        print("Loading encoder...")
        self.encoder = SparseEncoder(config).to(config.device).to(config.dtype)
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=config.device, weights_only=True)
        )
        self.encoder.eval()

        # Load decoder with LoRA
        print("Loading decoder...")
        self.decoder = PCDDecoder(config)
        # Load the fine-tuned LoRA weights
        self.decoder.model = PeftModel.from_pretrained(
            self.decoder.model.get_base_model(),
            decoder_lora_path,
            torch_dtype=config.dtype,
        ).to(config.device)
        self.decoder.model.eval()

        # Load feature labels if available
        self.feature_labels = {}
        if feature_labels_path and os.path.exists(feature_labels_path):
            with open(feature_labels_path) as f:
                raw = json.load(f)
            for k, v in raw.items():
                self.feature_labels[int(k)] = v.get("label", "unlabelled")
            print(f"Loaded {len(self.feature_labels)} feature labels")

    def get_labeled_concepts(
        self, top_idx: list[list[int]], top_vals: list[list[float]], position: int = 0
    ) -> list[dict]:
        """Return labeled concept info for a given token position.

        Args:
            top_idx: nested list [seq_len][k] of concept indices
            top_vals: nested list [seq_len][k] of concept values
            position: which token position to inspect (default: 0)

        Returns:
            List of dicts with 'concept_id', 'label', 'activation'
        """
        indices = top_idx[position] if position < len(top_idx) else []
        values = top_vals[position] if position < len(top_vals) else []

        results = []
        for idx, val in zip(indices, values):
            label = self.feature_labels.get(idx, "unlabelled")
            results.append({
                "concept_id": idx,
                "label": label,
                "activation": val,
            })
        return results

    def _encode_input(self, input_text: str):
        """Tokenize, run subject model, and encode activations.

        No padding. Feed the raw tokens to the subject model and extract
        the last middle_len positions (or all tokens if shorter). This
        avoids any train/inference distribution mismatch from padding.
        """
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        n_tokens = len(input_ids)

        input_tensor = torch.tensor([input_ids], device=self.config.device)

        # Extract the last middle_len positions as "middle", everything
        # before that is treated as prefix context.
        middle_len = min(n_tokens, self.config.middle_len)
        prefix_len = n_tokens - middle_len

        activations = self.subject.get_middle_activations(
            input_tensor, prefix_len, middle_len,
        )
        encoded, enc_info = self.encoder(activations)
        top_vals, top_idx = self.encoder.get_top_concepts(activations)

        return encoded, enc_info, top_vals, top_idx

    @torch.no_grad()
    def __call__(
        self,
        input_text: str,
        question: str = "What is the subject model thinking about?",
        max_new_tokens: int | None = None,
    ) -> dict:
        """Run PCD on input text.

        For pretrained-only models (no QA finetuning), the decoder generates
        text continuations from the encoded activations. These continuations
        reveal what information the encoder captured from the subject model.

        For finetuned models, the decoder can answer the question directly.

        Args:
            input_text: Text to analyze via the subject model
            question: Prompt/question for the decoder (used as prefix for generation)
            max_new_tokens: Override for generation length

        Returns:
            dict with 'pcd_continuation', 'subject_response', 'top_concepts'
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        encoded, enc_info, top_vals, top_idx = self._encode_input(input_text)

        # Mode 1: Pure continuation (what the pretrained decoder was trained to do)
        # Just give the soft tokens and let the decoder generate what comes next
        pcd_continuation = self.decoder.generate_from_soft_tokens(
            encoded,
            prompt_ids=None,
            max_new_tokens=max_new_tokens,
        )

        # Mode 2: Prompted continuation with a short probe
        # Give soft tokens + a short probe phrase and see what the decoder generates
        probes = [
            "The text discusses",
            "This passage is about",
            "The main topic is",
        ]
        probe_outputs = {}
        for probe in probes:
            probe_ids = self.tokenizer.encode(probe, add_special_tokens=False)
            probe_tensor = torch.tensor([probe_ids], device=self.config.device)
            output = self.decoder.generate_from_soft_tokens(
                encoded, probe_tensor, max_new_tokens=64,
            )
            probe_outputs[probe] = output[0]

        # Get subject model's direct response for comparison
        full_input = self.tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
        if full_input.ndim == 1:
            full_input = full_input.unsqueeze(0)
        full_input = full_input.to(self.config.device)

        subject_gen = self.subject.generate(full_input, max_new_tokens=max_new_tokens)
        subject_response = self.tokenizer.decode(
            subject_gen[0][full_input.shape[1]:], skip_special_tokens=True
        )

        return {
            "pcd_continuation": pcd_continuation[0],
            "probe_outputs": probe_outputs,
            "subject_response": subject_response,
            "top_concept_indices": top_idx[0].cpu().tolist(),
            "top_concept_values": top_vals[0].cpu().tolist(),
            "n_active_concepts": enc_info["n_active_concepts"],
        }

    @torch.no_grad()
    def ask(
        self,
        input_text: str,
        question: str,
        max_new_tokens: int = 32,
    ) -> dict:
        """Ask a question about the subject model's activations (post-finetuning).

        After QA fine-tuning, the decoder can answer structured questions about
        what the encoder captured from the subject model's internal state.

        Args:
            input_text: Text the subject model processes
            question: Natural language question about the text/model behavior
            max_new_tokens: Max tokens to generate for the answer

        Returns:
            dict with 'answer', 'question', 'top_concepts'
        """
        encoded, enc_info, top_vals, top_idx = self._encode_input(input_text)

        # Feed [soft_tokens] + [question] and generate the answer
        q_ids = self.tokenizer.encode(question, add_special_tokens=False)
        q_tensor = torch.tensor([q_ids], device=self.config.device)

        answer = self.decoder.generate_from_soft_tokens(
            encoded, q_tensor, max_new_tokens=max_new_tokens,
        )

        return {
            "question": question,
            "answer": answer[0],
            "top_concept_indices": top_idx[0].cpu().tolist(),
            "top_concept_values": top_vals[0].cpu().tolist(),
            "n_active_concepts": enc_info["n_active_concepts"],
        }

    @torch.no_grad()
    def ask_multiple(
        self,
        input_text: str,
        questions: list[str] | None = None,
        max_new_tokens: int = 32,
    ) -> list[dict]:
        """Ask multiple questions about the same input (efficient, single encode).

        Args:
            input_text: Text the subject model processes
            questions: List of questions. If None, uses default probing questions.
            max_new_tokens: Max tokens per answer

        Returns:
            List of dicts with 'question' and 'answer'
        """
        if questions is None:
            questions = [
                "What is the primary topic of the text?\nA. Science\nB. Technology\nC. Politics\nD. Other\nAnswer:",
                "What is the tone or sentiment of the text?\nA. Positive\nB. Negative\nC. Neutral\nD. Mixed\nAnswer:",
                "What domain does the text belong to?\nA. News\nB. Academic\nC. Blog\nD. Technical documentation\nAnswer:",
                "Is the text formal or informal?\nA. Formal\nB. Informal\nC. Semi-formal\nD. Cannot determine\nAnswer:",
            ]

        encoded, enc_info, top_vals, top_idx = self._encode_input(input_text)

        results = []
        for question in questions:
            q_ids = self.tokenizer.encode(question, add_special_tokens=False)
            q_tensor = torch.tensor([q_ids], device=self.config.device)

            answer = self.decoder.generate_from_soft_tokens(
                encoded, q_tensor, max_new_tokens=max_new_tokens,
            )
            results.append({
                "question": question,
                "answer": answer[0],
            })

        return results
