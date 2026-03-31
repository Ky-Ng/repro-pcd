from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pcd_config import PCDConfig

from jaxtyping import Float, Int
from torch import Tensor

class DecoderModel(nn.Module):
    def __init__(self, config: PCDConfig):
        super().__init__()

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=config.dtype,   
        ).to(config.device)

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(base_model, lora_config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward_train(self, 
                soft_token_act: Float[Tensor, "batch seq_soft"],
                suffix_ids: Int[Tensor, "batch seq_suffix"]
        ) -> Float[Tensor, "batch seq n_vocab"]:
        """
        Trains the decoder to predict the suffix_ids based on sparse activations from the Encoder for completions
        
        Visually:
            [Dummy tokens  ] + [suffix_ids]
                    |               |
        embed       |               |
                    x               |
                  patch             |
                    |               |
                    v               v 
            [soft_token_act] + [suffix_embed]

        Args:
            soft_token_act (Float[Tensor, "batch seq_soft"]): 
                Sparse activtions from Subject Model's l_read residuals passed through the Encoder. 
                We create dummy tokens of shape [batch, seq_soft] which are patched by these activations

            suffix_ids (Int[Tensor, "batch seq_suffix"]):
                Completion tokens to compute loss on            
        """
        model_token_embedding = self.model.get_input_embeddings()
        suffix_embeds = model_token_embedding(suffix_ids[:, :-1]) # [batch, seq_suffix-1, d_model], note we have no training signal for last token

        inputs_embed = torch.cat([soft_token_act, suffix_embeds], dim=1) # [batch, seq_soft + seq_suffix-1, d_model]

        B, total_len, _ = inputs_embed.shape
        _, n_soft, _ = soft_token_act.shape

        attention_mask = torch.ones([B, total_len], device=inputs_embed.device, dtype=torch.long)

        output = self.model(
            inputs_embeds=inputs_embed,
            attention_mask=attention_mask
        )

        # Compute loss on suffix tokens
        # Token Position (logit):   [soft_n-1, suffix_0, ..., suffix_n-2]
        # Suffix Label          :   [suffix_0, suffix_1, ..., suffix_n-1]      

        logits = output.logits # [batch, total_len, n_vocab]
        suffix_logits = logits[:, n_soft-1:, :] # [batch, n_suffix, n_vocab]
        targets = suffix_ids
        _, _, n_vocab = suffix_logits.shape

        # Flatten and calculate loss
        loss = nn.functional.cross_entropy(
            suffix_logits.reshape(-1, n_vocab), # [batch * n_suffix, n_vocab]
            targets.reshape(-1), # [batch * n_suffix]
            ignore_index=self.tokenizer.pad_token_id 
        )

        return loss
        