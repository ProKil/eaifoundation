from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.embodiedai.aux_losses.losses import AuxiliaryLoss

from qa_task.distributions import JointDistr


class PassThroughAuxLoss(AuxiliaryLoss):
    def get_aux_loss(self, **kwargs):
        pass

    def loss(
        self,
        step_count,
        batch,
        actor_critic_output: ActorCriticOutput[JointDistr],
        *args,
        **kwargs,
    ):
        return (actor_critic_output.extras["aux_loss"], {})
