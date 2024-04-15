from typing import Any, Dict, List, Sequence, Union

import gym
import torch
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
from gym.spaces import Discrete, MultiDiscrete, Tuple
from projects.objectnav_baselines.experiments.clip.mixins import (
    ClipResNetPreprocessGRUActorCriticMixin,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.mixins import ObjectNavPPOMixin
from relationship_graph.attrs_relations import RelationshipEnum
from transformers import T5Tokenizer

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
from qa_task.embodied_qa_model import EmbodiedQAActorCritic
from qa_task.qa_losses import PassThroughAuxLoss
from qa_task.qa_task import QATask
from qa_task.qa_task_sampler import QATaskSampler


class AuxObservationsProcessor(Preprocessor):
    """Represents a preprocessor that transforms data from a sensor or another
    preprocessor to the input of agents or other preprocessors. The user of
    this class needs to implement the process method and the user is also
    required to set the below attributes:

    # Attributes:
        input_uuids : List of input universally unique ids.
        uuid : Universally unique id.
        observation_space : ``gym.Space`` object corresponding to processed observation spaces.
    """

    input_uuids: List[str]
    uuid: str
    observation_space: gym.Space

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        observation_space: gym.Space,
        **kwargs: Any
    ) -> None:
        assert len(input_uuids) == 1, "Only one input is supported"
        self.uuid = output_uuid
        self.input_uuids = input_uuids
        self.observation_space = observation_space
        self.device = kwargs.get("device", torch.device("cpu"))

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Returns processed observations from sensors or other preprocessors.

        # Parameters

        obs : Dict with available observations and processed observations.

        # Returns

        Processed observation.
        """
        return {
            key: obs[self.input_uuids[0]][key].to(self.device)
            for key in obs[self.input_uuids[0]]
        }

    def to(self, device: torch.device) -> "Preprocessor":
        self.device = device
        return self


class QAConfig(BaseConfig):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    CLIP_MODEL_TYPE = "RN50"

    ADVANCE_SCENE_ROLLOUT_PERIOD = 50

    num_qa_pairs = 10
    max_answer_length = 32
    uvk_pickle_file = "/home/haoz/eaifoundation/exp/generate_probing_dataset/high_frequency_relationship.pkl"
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    DEFAULT_USE_WEB_RENDER = True

    ACTION_SPACE = Tuple(
        [
            BaseConfig.ACTION_SPACE,
            MultiDiscrete(
                [tokenizer.vocab_size] * num_qa_pairs * max_answer_length
            ),
        ]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.preprocessing_and_model = ClipResNetPreprocessGRUActorCriticMixin(
            sensors=self.sensors(),
            clip_model_type=self.CLIP_MODEL_TYPE,
            screen_size=self.SCREEN_SIZE,
            goal_sensor_type=None,
        )
        self.auxiliary_uuids = []

        self.extra_losses = {"qa_loss": (PassThroughAuxLoss("aux_loss"), 1)}

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
            AgentCameraStateSensor(uuid="agent_camera_state"),
            AgentArmStateSensor(uuid="agent_arm_state"),
        ]

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        return ObjectNavPPOMixin.training_pipeline(
            auxiliary_uuids=self.auxiliary_uuids,
            multiple_beliefs=False,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            num_steps=128,
            extra_losses=self.extra_losses,
            normalize_advantage=False,
            anneal_lr=False,
        )

    def preprocessors(
        self,
    ) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        _preprocessors = self.preprocessing_and_model.preprocessors()
        _preprocessors.append(
            AuxObservationsProcessor(
                input_uuids=["aux_observations"],
                output_uuid="aux_observations_output",
                observation_space=MultiDiscrete([self.num_qa_pairs, 512]),
            )
        )
        return _preprocessors

    def create_model(self, **kwargs) -> nn.Module:
        return EmbodiedQAActorCritic(
            action_space=self.ACTION_SPACE,
            observation_space=kwargs[
                "sensor_preprocessor_graph"
            ].observation_spaces,
            obs_sensor_uuids=[
                "rgb_clip_resnet",
                "agent_camera_state",
                "agent_arm_state",
            ],
            t5_model_path="/home/haoz/t5x-frozen-lm-qa-base",
            hidden_size=512,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs):
        kwargs["num_qa_pairs"] = cls.num_qa_pairs
        kwargs["uvk_pickle_file"] = cls.uvk_pickle_file
        kwargs["tokenizer"] = cls.tokenizer
        return QATaskSampler(**kwargs)

    def tag(self):
        return "QATask-ClipRN50-GRU-DDPPO"

    def controller_kwargs(self):
        ret = super().controller_kwargs()
        ret["branch"] = "nanna-grasp-force"
        return ret
