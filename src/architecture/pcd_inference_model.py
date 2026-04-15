import torch

from src.architecture.decoder_model import DecoderModel
from src.architecture.sparse_encoder import SparseEncoder
from src.architecture.subject_model import SubjectModel
from src.pcd_config import PCDConfig
from src.training.utils import load_checkpoint as _load_from_disk


class PCDInferenceModel:
    def __init__(self, config: PCDConfig):
        self.config = config
        self.subject = SubjectModel(config)
        self.encoder = SparseEncoder(config).to(config.device).eval()
        self.decoder = DecoderModel(config).to(config.device).eval()

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        _load_from_disk(self.encoder, self.decoder,
                        checkpoint_dir, self.config.device)

    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        decoder_question: str | None = None, 
        max_new_tokens: int = 256
    ) -> str:
        """
        Decoder questions are always wrapped in a Chat Template

        Note: at this time generate only supports one prompt at a time
        TODO: support batched generation by handling token index to account for padding
        """
        # 1. Chat-template and tokenize the prompt via subject tokenizer
        templated = self.subject.apply_chat_template(prompt)
        enc = self.subject.tokenize(templated)
        tokens = enc["input_ids"]                 # [1, seq]
        attn = enc["attention_mask"]            # [1, seq]

        # 2. Extract residual-stream activations at l_read over the full user span.
        #    Deployment case in subject_model.py — no fixed prefix/middle/suffix split.
        seq_len = tokens.shape[1]
        activations = self.subject.get_middle_activations(
            tokens=tokens,
            attention_mask=attn,
            start_extract=0,
            end_extract=seq_len,
        )  # [1, seq, d_model]

        # 3. Encode → sparse re-embedding (soft tokens for the decoder).
        with torch.autocast(device_type=self.config.device, dtype=self.config.dtype):
            sparse_embedding, _info = self.encoder(activations)

        # 4. Add decoder question as context_ids if present
        context_ids = None
        if decoder_question:
            decoder_question_with_prompt = self.decoder.apply_chat_template(decoder_question)
            tokenized_context = self.decoder.tokenize(decoder_question_with_prompt)
            context_ids = tokenized_context["input_ids"]

        # 5. Generate from decoder.
        outputs = self.decoder.generate(
            soft_token_acts=sparse_embedding,
            context_ids=context_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        return outputs[0]
