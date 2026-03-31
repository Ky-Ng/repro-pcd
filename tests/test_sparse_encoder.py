import torch

from src.pcd_config import PCDConfig
from src.sparse_encoder import SparseEncoder


config = PCDConfig()
my_sparse_encoder = SparseEncoder(config)

batch, seq, d_model = 2, 10, config.d_model
sample_activations = torch.rand([batch, seq, d_model])

encoder_out, info = my_sparse_encoder(sample_activations)

print("Encoder Output Shape", encoder_out.shape)
print("Info", info)