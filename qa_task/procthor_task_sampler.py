"""General ProcTHOR task sampler. Adapted from Luca's boolset code.
"""

import copy
import itertools
import random
import warnings
from collections import defaultdict
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Union

import ai2thor
import ai2thor.server
import attr
import canonicaljson
import gym
import numpy as np
from ai2thor.controller import Controller
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.misc_utils import (
    md5_hash_str_as_int,
    prepare_locals_for_super,
)
from allenact.utils.system import get_logger
from allenact_plugins.manipulathor_plugin.manipulathor_constants import (
    ADDITIONAL_ARM_ARGS,
    DONE,
    DROP,
    LOOK_DOWN,
    LOOK_UP,
    MOVE_AHEAD,
    MOVE_ARM_HEIGHT_M,
    MOVE_ARM_HEIGHT_P,
    MOVE_ARM_X_M,
    MOVE_ARM_X_P,
    MOVE_ARM_Y_M,
    MOVE_ARM_Y_P,
    MOVE_ARM_Z_M,
    MOVE_ARM_Z_P,
    MOVE_BACK,
    PICKUP,
    ROTATE_ELBOW_M,
    ROTATE_ELBOW_P,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    ROTATE_WRIST_PITCH_M,
    ROTATE_WRIST_PITCH_P,
    ROTATE_WRIST_ROLL_M,
    ROTATE_WRIST_ROLL_P,
    ROTATE_WRIST_YAW_M,
    ROTATE_WRIST_YAW_P,
)
from relationship_graph.attrs_relations import (
    EntityPairEnum,
    NodeTypeEnum,
    RelationshipEnum,
)
from relationship_graph.graphs import (
    StrictMultiDiGraph,
    build_relationship_graph,
    build_soft_relationship_graph,
)

from boolset.tasks_and_samplers import (
    AgentPose,
    HouseAugmenter,
    ProcTHORDataset,
    Vector3,
)


class ProcTHORTaskSampler(TaskSampler):
    def __init__(
        self,
        procthor_dataset: ProcTHORDataset,
        sensors: Sequence[Sensor],
        house_repeats: Union[int, float],
        max_steps: int,
        reset_on_scene_replay: bool,
        augmenter: Optional[HouseAugmenter] = None,
        controller_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        allow_house_skip: bool = True,
        **kwargs,
    ):
        assert len(kwargs) == 0 or (
            len(kwargs) == 1 and "mp_ctx" in kwargs
        ), f"{kwargs.keys()}"
        assert house_repeats > 0 and (
            isinstance(house_repeats, int) or house_repeats == float("inf")
        )

        self.procthor_dataset = procthor_dataset
        self.sensors = sensors
        self.house_repeats = house_repeats
        self.augmenter = augmenter
        self.max_steps = max_steps
        self.reset_on_scene_replay = reset_on_scene_replay

        self.controller_kwargs = dict(
            gridSize=0.25,
            width=224,
            height=224,
            visibilityDistance=1.0,
            agentMode="arm",
            fieldOfView=100,
            server_class=ai2thor.fifo_server.FifoServer,
            useMassThreshold=True,
            massThreshold=10,
            autoSimulation=False,
            autoSyncTransforms=True,
        )
        self.controller_kwargs.update(controller_kwargs)  # type: ignore

        self.allow_house_skip = allow_house_skip

        self._seed = (
            seed if seed is not None else random.randint(0, 2**31 - 1)
        )
        self._random: Optional[random.Random] = None
        self._controller: Optional[Controller] = None
        self._house_order: Optional[List[int]] = None

        self._remaining_inds: List[int] = []
        self._current_house: Optional[Dict[str, Any]] = None
        self._current_house_ind: Optional[int] = None
        self._current_repeat_ind = 0

        self.reset()

    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._random = random.Random(self._seed)
        self._np_random = np.random.RandomState(self._seed)

    def last_sampled_task(self) -> Optional[Task]:
        return self._last_sampled_task

    def reset(self) -> None:
        self._random = random.Random(self._seed)
        self._np_random = np.random.RandomState(self._seed)
        self._refresh_remaining_inds()
        self._current_repeat_ind = 0
        self._last_sampled_task = None

    @property
    def seed(self):
        return self._seed

    @property
    def controller(self) -> Controller:
        if self._controller is None:
            self._controller = Controller(**self.controller_kwargs)
        return self._controller

    @property
    def length(self) -> Union[int, float]:
        return float("inf")

    def close(self) -> None:
        if self._controller is not None:
            self._controller.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        return True

    def _refresh_remaining_inds(self) -> None:
        self._remaining_inds = [
            int(i)
            for i in self._np_random.permutation(len(self.procthor_dataset))
        ]

    def _setup_next_house(self, controller: Controller):
        for _ in range(len(self.procthor_dataset)):
            if len(self._remaining_inds) == 0:
                self._refresh_remaining_inds()
            self._current_house_ind = self._remaining_inds.pop()
            self._current_house = self.procthor_dataset.initialize_house(
                controller=controller, index=self._current_house_ind
            )

            if self._current_house is not None:
                break
            elif not self.allow_house_skip:
                raise RuntimeError(
                    f"Could not set up house with index {self._current_house_ind}"
                )

        if self._current_house is None:
            raise RuntimeError("Could not set up a house.")

    def _reset_state(self, force_advance_scene):
        if (
            force_advance_scene
            or self._last_sampled_task is None
            or self._current_repeat_ind >= self.house_repeats
        ):
            self._setup_next_house(controller=self.controller)
        else:
            if self.reset_on_scene_replay:
                self.controller.reset()
                self.procthor_dataset.initialize_house(
                    controller=self.controller, index=self._current_house_ind
                )
            else:
                self.controller.step(
                    action="ResetObjectFilter",
                    raise_for_failure=True,
                )

        self._current_repeat_ind += 1

    def _random_agent_pose(self, n_retries: int = 10):
        if self._current_house_ind is None:
            raise Exception(
                "reset() must be called before randomizing agent pose"
            )
        for i in range(n_retries):
            standing = (
                {}
                if self.controller.initialization_parameters["agentMode"]
                == "locobot"
                else {"standing": self._np_random.choice([False, True])}
            )
            starting_pose: AgentPose = {
                "position": self._np_random.choice(
                    self.procthor_dataset.reachable_positions_map[  # type: ignore
                        self._current_house_ind
                    ]
                ),
                "rotation": Vector3(
                    x=0, y=self._np_random.random() * 360, z=0
                ),
                "horizon": self._np_random.randint(-1, 2) * 30,
                "standing": standing["standing"]
                if "standing" in standing
                else None,
            }

            md = self.controller.step(
                action="TeleportFull", **starting_pose
            ).metadata
            if not md["lastActionSuccess"]:
                if i == n_retries - 1:
                    warnings.warn(
                        f"Teleport failed in {self._current_house_ind} {n_retries} times!"
                    )
                continue
            break
        else:
            raise Exception("Could not find a valid starting pose.")

        return starting_pose
