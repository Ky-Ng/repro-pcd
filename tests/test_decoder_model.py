import torch

from src.decoder_model import DecoderModel
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

pretrain_loss = my_decoder_model.forward_train(soft_token_acts, suffix_ids)

print("Soft Token Activations shape", soft_token_acts.shape)
print("Suffix IDs shape", suffix_ids.shape)
print("pretraining loss on sample soft tokens acts and suffix ids", pretrain_loss)
