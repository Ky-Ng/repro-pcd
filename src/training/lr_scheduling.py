import math
from typing import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def _step_cosine_schedule_with_warmup(
    warmup_steps: int,
    total_steps: int
) -> Callable[[int], float]:
    """
    Cosine Scheduling with Linear Warmup, take step (int) and returns the learning rate (float)

    Linear warmup till warmup_steps

    After warmup use cosine scheduling:
    $0.5 * (1 + cos(t))$

        - cos(t) goes from [-1, 1]
        - 1 + cos(t) goes from [0, 2]
        - 0.5 * (1 + cos(t)) goes from [0, 1]
    """

    def lr_lambda(step: int) -> float:
        # Linear warmup
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        
        # Cosine decay [1, 0]
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return lr_lambda

def get_optimizer_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int
) -> LambdaLR:
    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=_get_cosine_schedule_with_warmup(warmup_steps, total_steps)
    )