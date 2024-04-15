from typing import Any, Optional, Sequence

import gym
import torch
from ai2thor.controller import Controller
from allenact.base_abstractions.sensor import Sensor
from allenact_plugins.manipulathor_plugin.arm_calculation_utils import (
    coord_system_transform,
    world_coords_to_agent_coords,
)
from allenact_plugins.manipulathor_plugin.manipulathor_environment import (
    ManipulaTHOREnvironment,
)

from boolset.tasks_and_samplers import BoolSetTask


class BoolSetTaskCompleteSensor(Sensor):
    def __init__(self, uuid: str):
        super().__init__(
            uuid=uuid, observation_space=gym.spaces.Box(0, 1, shape=(1,))
        )

    def get_observation(
        self,
        env: Controller,
        task: Optional[BoolSetTask],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return [1.0 * task.last_energy == 0.0]


class RelationshipsSensor(Sensor[Controller, BoolSetTask]):
    def __init__(
        self,
        uuid: str,
        entities: Sequence[str],
        relationship_types: Sequence[str],
        max_relationships: int,
        max_instances: int,
        **kwargs: Any,
    ):
        super().__init__(
            uuid=uuid,
            observation_space=gym.spaces.Dict(
                {
                    "e0": gym.spaces.MultiDiscrete(
                        [len(entities)] * max_relationships
                    ),
                    "e1": gym.spaces.MultiDiscrete(
                        [len(entities)] * max_relationships
                    ),
                    "i0": gym.spaces.MultiDiscrete(
                        [max_instances] * max_relationships
                    ),
                    "i1": gym.spaces.MultiDiscrete(
                        [max_instances] * max_relationships
                    ),
                    "relationship_type": gym.spaces.MultiDiscrete(
                        [len(relationship_types)] * max_relationships
                    ),
                }
            ),
            **kwargs,
        )
        entities = [e.lower() for e in entities]
        assert entities == sorted(entities)

        relationship_types = [rt.lower() for rt in relationship_types]
        assert relationship_types == sorted(relationship_types)

        self.entity_to_index = {e: i for i, e in enumerate(entities)}
        self.relationship_type_to_index = {
            rt: i for i, rt in enumerate(relationship_types)
        }

    def get_observation(
        self,
        env: Controller,
        task: Optional[BoolSetTask],
        *args: Any,
        **kwargs: Any,
    ) -> Any:

        assert task is not None

        e0 = self.entity_to_index[task.entity_0.lower()]
        e1 = self.entity_to_index[task.entity_1.lower()]
        i0 = 0
        i1 = 0
        rel = self.relationship_type_to_index[
            task.relationship_type.name.lower()
        ]

        return {
            "e0": [e0],
            "e1": [e1],
            "i0": [i0],
            "i1": [i1],
            "relationship_type": [rel],
        }


class AgentCameraStateSensor(Sensor[Controller, BoolSetTask]):
    def __init__(self, uuid: str, **kwargs: Any):
        super().__init__(
            uuid=uuid,
            observation_space=gym.spaces.MultiDiscrete([1] * 5),
            **kwargs,
        )

    def get_observation(
        self,
        env: Controller,
        task: Optional[BoolSetTask],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        md = env.last_event.metadata
        agent = md["agent"]

        position_list = [0] * 5
        position_list[0] = 1 * agent["isStanding"]

        height = 1 + round(agent["cameraHorizon"] / 30)
        assert 0 <= height <= 3
        position_list[height] = 1

        return position_list


class AgentArmStateSensor(Sensor[Controller, BoolSetTask]):
    def __init__(self, uuid: str, **kwargs: Any):
        super().__init__(
            uuid=uuid,
            observation_space=gym.spaces.Box(low=-5, high=5, shape=(12,)),
            **kwargs,
        )

    def get_observation(
        self,
        env: Controller,
        task: Optional[BoolSetTask],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        md = env.last_event.metadata

        agent = md["agent"]
        arm = md["arm"]
        joints = arm["joints"]

        joint_world_positions = [
            ManipulaTHOREnvironment.correct_nan_inf(j["position"])
            for j in joints
        ]

        null_rot = {"x": 0, "y": 0, "z": 0}
        joint_rel_positions = [
            world_coords_to_agent_coords(
                world_obj={"position": jp, "rotation": null_rot},
                agent_state=agent,
                use_cache=False,
            )["position"]
            for jp in joint_world_positions
        ]

        joint_rel_polar_positions = [
            coord_system_transform(position=jrp, coord_system="polar_radian")
            for jrp in joint_rel_positions
        ]

        return torch.cat(joint_rel_polar_positions, dim=-1).numpy()
