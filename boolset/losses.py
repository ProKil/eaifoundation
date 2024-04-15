from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput


class ShouldTakeDoneLoss(AbstractActorCriticLoss):
    def __init__(self, target_uuid: str, done_index: int):
        self.target_uuid = target_uuid
        self.done_index = done_index

    def loss(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, float]],
        Tuple[torch.Tensor, Dict[str, float], Dict[str, float]],
    ]:

        loss = F.binary_cross_entropy(
            actor_critic_output.distributions.probs_tensor[
                ..., self.done_index
            ],
            batch["observations"][self.target_uuid].squeeze(-1),
        )
        return loss, {
            "binary_ce": loss.item(),
        }
