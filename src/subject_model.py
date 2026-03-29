from src.pcd_config import PCDConfig
from transformers import AutoTokenizer

class SubjectModel:
    """
    Frozen Subject Model where we extract activations from l_read with a forward hook
    """
    def __init__(self, config: PCDConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    def generate(self, x):
        return self.tokenizer(x)
    
