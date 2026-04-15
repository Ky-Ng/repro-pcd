from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pcd_config import PCDConfig

from jaxtyping import Float, Int, Bool
from torch import Tensor

IGNORE_INDEX = -100

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
    
    def forward_train(
        self,
        soft_token_acts: Float[Tensor, "batch n_soft d_model"],
        target_ids: Int[Tensor, "batch n_target"],
        context_ids: Int[Tensor, "batch n_context"] | None = None,
        soft_token_mask: Bool[Tensor, "batch n_soft"] | None = None
    ) -> Float[Tensor, ""]:
        """
        Args:
            soft_token_acts (Float[Tensor, "batch n_soft d_model"]): 
                activations from the Encoder, patched in as the initial soft tokens
            target_ids (Int[Tensor, "batch n_target"]):
                tokens to be predicted
            context_ids (Int[Tensor, "batch n_context"]):
                Optional query/context token IDs placed between soft tokens and targets.
                Right-padded with pad_token_id when batching variable-length sequences. 
                Not used during pretraining.
                Loss is not calculated on these tokens
            soft_token_mask (Bool[Tensor, "batch n_soft d_model"]):
                Optional token mask for soft_token_acts
        
        Returns:
            loss (Float[Tensor, ""]): Scalar Tensor representing the loss

        Trains the decoder to predict the `target_ids` from `soft_tokens` (and `context_ids` if `context_ids` is present).

        Note: `context_ids` are present only during Finetuning on SynthSys but not for FineWeb pretraining

        Visual for Pretraining FineWeb Completions:
            [Dummy tokens  ] + [target_ids]
                    |               |
        embed       |               |
                    x               |
                  patch             |
                    |               |
                    v               v 
            [soft_token_act] + [suffix_embed]
                                ^Loss calculated only on targets

            - In FineWeb completions, `soft_token_acts` are of length `n_middle`, `target_ids` are of length `n_suffix` (convention in the paper)
            - These lengths are always fixed, no padding needed
        
        Visual for Finetune SynthSys:
            [Dummy tokens  ] + [query_token_ids] + [target_ids (single letter for MCQ)]
                    |               |                   |
        embed       |               |                   |
                    x               |                   |
                  patch             |                   |
                    |               |                   |
                    v               v                   v
            [soft_token_act] + [query_embeds]    + [suffix_embed]
                    *                                   ^Loss calculated only on target (single MCQ letter answer)
            [soft_token_mask]

            - In SynthSys examples, `soft_token_acts` are the user query (starting after "<im_start>user") and are of variable length, soft_token_mask should be used         
            - `soft_token_mask` will be used to mask out activations that do not to be used
            - `target_ids` will be a single letter representing the MCQ token (though may change based on the finetuning dataset)
            - `context_ids` will be right-padded with pad_token_id
        """
        B, n_soft, _d = soft_token_acts.shape
        n_target = target_ids.shape[1]
        token_embedding_layer = self.model.get_input_embeddings()

        # Step 1) Build input to the model
        embed_parts = [soft_token_acts]
        mask_parts = [soft_token_mask if soft_token_mask is not None 
                            else torch.ones(B, n_soft, 
                                       device=soft_token_acts.device, 
                                       dtype=torch.long
                            )
                        ]
        input_len = n_soft

        # Optionally add in context tokens if provided
        if context_ids is not None:
            context_embeds = token_embedding_layer(context_ids)
            embed_parts.append(context_embeds)
            mask_parts.append(
                (context_ids != self.tokenizer.pad_token_id).long()
            )
            input_len += context_embeds.shape[1]

        # Target embedding tokens
        # subtract 1 since we don't have a ground truth for the last token  
        target_embeds = token_embedding_layer(target_ids[:, :-1]) # [B, n_targets-1]
        embed_parts.append(target_embeds)

        # note: target_embeds is only a single token for SynthSys but we keep the implementation extensible
        mask_parts.append(
            (target_ids[:, :n_target-1] != self.tokenizer.pad_token_id).long() 
        )
        
        # Prep inputs to model
        input_embeds = torch.cat(embed_parts, dim=1) # [B, total_len, d_model]
        attention_mask = torch.cat(mask_parts, dim=1) # [B, total_len]

        # Step 2) Extract logits from the model
        output = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        logits = output.logits # [B, total_len, n_vocab]

        # Step 3) Calculate loss on the logits
        # logits             = [prefix_0, prefix_1, ... prefix_n-1, target_0, ...]
        # logits predictions = [prefix_1, prefix_2, ... target_0,   target_1, ...]
        target_logits = logits[:, input_len-1:, :] # [B, n_target, n_vocab]
        labels = target_ids.clone()
        
        # Note: we use IGNORE_INDEX instead of pad token in case we using eos as pad
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        
        loss = nn.functional.cross_entropy(
            target_logits.reshape(-1, target_logits.size(-1)), # [B * n_target, n_vocab]
            labels.reshape(-1),                               # [B * n_target]  
            ignore_index=IGNORE_INDEX
        )

        return loss

    def forward_inference(
        soft_token_acts: Float[Tensor, "batch n_soft d_model"],
        context_ids: Int[Tensor, "batch n_context"] | None = None,
        soft_token_mask: Bool[Tensor, "batch n_soft"] | None = None   
    ) -> Float[Tensor, "batch n_soft n_vocab"]:
        