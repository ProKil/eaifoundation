import warnings
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Set, Tuple, Union

import gym
import numpy as np
from ai2thor.controller import Controller
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.task import Task
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
from question_answering.question_generation import question_generation
from relationship_graph.attrs_relations import (
    EntityPairEnum,
    NodeTypeEnum,
    RelationshipEnum,
)

from qa_task.arm_utils import arm_agent_step
from qa_task.procthor_task_sampler import ProcTHORTaskSampler


class ProcTHORTask(Task):
    ACTIONS = (
        MOVE_AHEAD,
        MOVE_BACK,
        ROTATE_LEFT,
        ROTATE_RIGHT,
        MOVE_ARM_HEIGHT_P,
        MOVE_ARM_HEIGHT_M,
        MOVE_ARM_X_P,
        MOVE_ARM_X_M,
        MOVE_ARM_Y_P,
        MOVE_ARM_Y_M,
        MOVE_ARM_Z_P,
        MOVE_ARM_Z_M,
        ROTATE_WRIST_PITCH_P,
        ROTATE_WRIST_PITCH_M,
        ROTATE_WRIST_YAW_P,
        ROTATE_WRIST_YAW_M,
        ROTATE_WRIST_ROLL_P,
        ROTATE_WRIST_ROLL_M,
        ROTATE_ELBOW_P,
        ROTATE_ELBOW_M,
        LOOK_UP,
        LOOK_DOWN,
        DROP,
        PICKUP,
        DONE,
    )

    def __init__(
        self,
        env: Controller,
        sensors: Union[SensorSuite, Sequence[Sensor]],
        task_info: Dict[str, Any],
        max_steps: int,
        house: Dict[str, Any],
        **_kwargs,
    ):
        super(ProcTHORTask, self).__init__(env, sensors, task_info, max_steps)
        self.task_info = task_info
        self.house = house
        self.took_done_action = False
        self.seen_object_ids: Set[str] = set()
        self.visited_positions: Set[Tuple[int, int]] = set()

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(len(self.ACTIONS))

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def get_new_object_reward(self) -> float:
        visible_objects = [
            o["objectId"]
            for o in self.env.last_event.metadata["objects"]
            if o["visible"]
        ]
        reward: float = 0.0
        for object in visible_objects:
            if object not in self.seen_object_ids:
                self.seen_object_ids.add(object)
                reward += 0.1
        return reward

    def get_coverage_reward(self) -> float:
        discrete_position = (
            int(self.env.last_event.metadata["agent"]["position"]["x"] * 4),
            int(self.env.last_event.metadata["agent"]["position"]["y"] * 4),
        )
        if discrete_position not in self.visited_positions:
            self.visited_positions.add(discrete_position)
            return 0.1
        return 0.0

    def _step(self, action: int) -> RLStepResult:
        action = self.ACTIONS[action]
        if action == DONE:
            self.took_done_action = True
        else:
            arm_agent_step(controller=self.env, action=action)

        reward = self.get_new_object_reward() + self.get_coverage_reward()

        if np.isnan(reward):
            get_logger().error(
                f"NaN reward encountered!!! nError in {self.task_info}.\nMetadata: {self.env.last_event.metadata}."
            )
            reward = 0.0

        return RLStepResult(
            observation=self.get_observations(),
            reward=reward,
            done=self.is_done(),
            info=None,
        )

    def metrics(self) -> Dict[str, Any]:
        return {
            **super(ProcTHORTask, self).metrics(),
        }

    def reached_terminal_state(self) -> bool:
        return self.took_done_action

    def close(self) -> None:
        self.env.stop()
