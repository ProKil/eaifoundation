from typing import Tuple

import torch
from allenact.base_abstractions.distributions import Distr


class DeltaDistr(Distr):
    """δ distribution"""

    def __init__(self, target_value: torch.Tensor, inf_value: float = 1000):
        self.target_value = target_value
        self.inf_value = inf_value

    def entropy(self):
        return torch.tensor(0.0)

    def mode(self):
        return self.target_value

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        action_shape = actions.shape
        action_dims = len(action_shape)
        assert (
            action_shape[-action_dims:] == self.target_value.shape
        ), "The last dimensions of actions must match the shape of target_value"
        # matching all of the last dimensions of actions and target_value
        result = actions == self.target_value
        # contracting them to a single dimension
        for _ in range(action_dims):
            result = result.prod(-1)
        return result.float().clamp_min(1e-8).log()

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        shape = sample_shape + self.target_value.shape
        return self.target_value.expand(shape)


class JointDistr(Distr):
    """Joint distribution"""

    def __init__(self, distrs: Tuple[Distr]):
        self.distrs = distrs

    def entropy(self):
        return sum([distr.entropy() for distr in self.distrs])

    def mode(self):
        return tuple(distr.mode() for distr in self.distrs)

    def log_prob(self, actions: Tuple[torch.Tensor]) -> torch.Tensor:
        assert len(actions) == len(
            self.distrs
        ), "Number of actions must match number of distributions"
        return sum([distr.log_prob(action) for action, distr in zip(actions, self.distrs)])  # type: ignore

    def sample(self, sample_shape=torch.Size()) -> Tuple[torch.Tensor, ...]:
        return tuple(distr.sample(sample_shape) for distr in self.distrs)
