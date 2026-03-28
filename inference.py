"""End-to-end PCD inference pipeline."""

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
