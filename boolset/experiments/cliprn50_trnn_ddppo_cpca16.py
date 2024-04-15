from allenact.embodiedai.aux_losses.losses import CPCA16SoftMaxLoss

from boolset.experiments.cliprn50_trnn_ddppo import ClipRN50DDPPOConfig


class ClipRN50DDPPOCPCA16Config(ClipRN50DDPPOConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auxiliary_uuids = [CPCA16SoftMaxLoss.UUID]

    def tag(self):
        return "BoolSet-ClipRN50-GRU-DDPPO-CPCA16"
