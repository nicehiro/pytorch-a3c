import torch.optim as optim
from typing import Tuple


class SharedAdam(optim.Adam):
    def __init__(
        self,
        params,
        lr: float,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
