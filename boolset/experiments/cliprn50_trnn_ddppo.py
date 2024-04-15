from typing import Sequence, Union

import torch.nn as nn
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder, TrainingPipeline
from allenact_plugins.clip_plugin.clip_preprocessors import (
    ClipResNetPreprocessor,
)
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from allenact_plugins.manipulathor_plugin.manipulathor_constants import (
    DONE,
)
from projects.objectnav_baselines.experiments.clip.mixins import (
    ClipResNetPreprocessGRUActorCriticMixin,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.mixins import ObjectNavPPOMixin
from relationship_graph.attrs_relations import RelationshipEnum

from boolset.constants import OBJECT_TYPES
from boolset.experiments.base import BaseConfig
from boolset.losses import ShouldTakeDoneLoss
from boolset.models import RecurrentRelationshipTransformerActorCritic
from boolset.sensors import (
    AgentArmStateSensor,
    AgentCameraStateSensor,
    BoolSetTaskCompleteSensor,
    RelationshipsSensor,
)
from boolset.tasks_and_samplers import BoolSetTask


class ClipRN50DDPPOConfig(BaseConfig):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    CLIP_MODEL_TYPE = "RN50"

    ADVANCE_SCENE_ROLLOUT_PERIOD = 50

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.preprocessing_and_model = ClipResNetPreprocessGRUActorCriticMixin(
            sensors=self.sensors(),
            clip_model_type=self.CLIP_MODEL_TYPE,
            screen_size=self.SCREEN_SIZE,
            goal_sensor_type=None,
        )
        self.auxiliary_uuids = []

        self.extra_losses = {
            "take_done_loss": (
                ShouldTakeDoneLoss(
                    target_uuid=next(
                        s.uuid
                        for s in self.sensors()
                        if isinstance(s, BoolSetTaskCompleteSensor)
                    ),
                    done_index=BoolSetTask.ACTIONS.index(DONE),
                ),
                1.0,
            )
        }

    def sensors(self):
        return [
            RGBSensorThor(
                height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
                width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid="rgb_lowres",
                mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
                stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            ),
            RelationshipsSensor(
                uuid="relationship_sensor",
                entities=sorted(
                    [
                        e.lower()
                        for e in [
                            "Agent",
                            "Kitchen",
                            "Bathroom",
                            "Livingroom",
                            "Bedroom",
                            *OBJECT_TYPES,
                        ]
                    ]
                ),
                relationship_types=sorted(
                    [r.name.lower() for r in list(RelationshipEnum)]
                ),
                max_relationships=1,
                max_instances=1,
            ),
            BoolSetTaskCompleteSensor("task_complete"),
            AgentCameraStateSensor(uuid="agent_camera_state"),
            AgentArmStateSensor(uuid="agent_arm_state"),
        ]

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        return ObjectNavPPOMixin.training_pipeline(
            auxiliary_uuids=self.auxiliary_uuids,
            multiple_beliefs=False,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            num_steps=80,
            extra_losses=self.extra_losses,
            normalize_advantage=False,
            anneal_lr=False,
        )

    def preprocessors(
        self,
    ) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return self.preprocessing_and_model.preprocessors()

    def create_model(self, **kwargs) -> nn.Module:

        relationship_sensor_uuid = next(
            (
                s.uuid
                for s in self.sensors()
                if isinstance(s, RelationshipsSensor)
            )
        )

        return RecurrentRelationshipTransformerActorCritic(
            action_space=self.ACTION_SPACE,
            observation_space=kwargs[
                "sensor_preprocessor_graph"
            ].observation_spaces,
            obs_sensor_uuids=[
                "rgb_clip_resnet",
                "agent_camera_state",
                "agent_arm_state",
            ],
            relationship_sensor_uuid=relationship_sensor_uuid,
            obs_transformer_layers=2,
            rel_transformer_layers=2,
            memory_shape=(10, 768),
            auxiliary_uuids=self.auxiliary_uuids,
        )

    def tag(self):
        return "BoolSet-ClipRN50-GRU-DDPPO"
