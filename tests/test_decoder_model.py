import torch

from src.architecture.decoder_model import DecoderModel
from src.pcd_config import PCDConfig


config = PCDConfig()

my_decoder_model = DecoderModel(config)

batch = 2
soft_token_acts = torch.rand(
    [batch, config.n_middle, config.d_model], dtype=config.dtype, device=config.device
)
suffix_ids = torch.randint(
    low=0, high=config.n_vocab, size=[batch, config.n_suffix], device=config.device
)

print("Soft Token Activations shape", soft_token_acts.shape)
print("Suffix IDs shape", suffix_ids.shape)

print("#"*20, "Test Forward Train with Soft Tokens only", "#"*20)
pretrain_loss = my_decoder_model.forward_train(soft_token_acts, suffix_ids)
print("finetune loss on sample soft tokens acts", pretrain_loss)

print("#"*20, "Test Forward Train with Soft Tokens and Context Tokens", "#"*20)
print("pretraining loss on sample soft tokens acts and suffix ids", pretrain_loss)

context_ids = torch.randint(
     low=0, high=config.n_vocab, size=[batch, config.n_middle], device=config.device
)
fine_tune_loss = my_decoder_model.forward_train(
    soft_token_acts=soft_token_acts, 
    context_ids=context_ids,
    target_ids=suffix_ids,
    soft_token_mask=torch.ones(batch, config.n_prefix, device=config.device)
)

print("finetune loss on sample soft tokens acts and suffix ids", fine_tune_loss)

print("#"*20, "Test Generate Function", "#"*20)
output = my_decoder_model.generate(
    soft_token_acts=soft_token_acts,
    soft_token_mask=torch.ones([batch, soft_token_acts.shape[1]], dtype=torch.long, device=soft_token_acts.device),
    context_ids=context_ids,
    max_new_tokens=256,
    do_sample=False
)

for b in range(batch):
    print("Batch 1 Decode")
    print("Num Tokens: ", len(output[b]))

    print("<START DECODE>")
    print(output[b])
    print("<END DECODE>")

print("#"*20, "Test Generate Function on actual text", "#"*20)

prompt = "I am a woman. What should I wear to a wedding?"
prompt_with_chat_template = my_decoder_model.apply_chat_template(prompt)

print("Prompt", prompt)
print("Default Model Prompt with Chat Template", prompt_with_chat_template)

inputs = my_decoder_model.tokenize(prompt_with_chat_template)

input_tokens_ids = inputs["input_ids"]
print("Input Ids", input_tokens_ids)

dummy_soft_batch_size = 1
dummy_soft_token_acts = torch.rand(
    [dummy_soft_batch_size, config.n_middle, config.d_model], dtype=config.dtype, device=config.device
)

output = my_decoder_model.generate(
    soft_token_acts=dummy_soft_token_acts,
    soft_token_mask=torch.zeros([dummy_soft_batch_size, dummy_soft_token_acts.shape[1]], dtype=torch.long, device=soft_token_acts.device),
    context_ids=input_tokens_ids,
    max_new_tokens=256,
    do_sample=False
)

print("decoder outputs", output)