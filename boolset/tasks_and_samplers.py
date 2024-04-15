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
from procthor.utils.upgrade_house_version import HouseUpgradeManager
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

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

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


class Vector3(TypedDict):
    x: float
    y: float
    z: float


class AgentPose(TypedDict):
    position: Vector3
    rotation: Vector3
    horizon: int
    standing: Optional[bool]


def is_probability_validator(instance, attribute, value: float):
    assert isinstance(value, (float, int)) and 0 <= value <= 1


def arm_agent_step(
    controller: Controller,
    action: str,
    move_agent_dist: Optional[float] = None,
    rotate_degrees: Optional[float] = None,
    move_base_dist: float = 0.05,
    move_arm_dist: float = 0.05,
    rotate_wrist_degrees: float = 15,
    rotate_elbow_degrees: float = 15,
    simplify_physics: bool = False,
    render_image: bool = True,
) -> ai2thor.server.Event:
    """Take a step in the ai2thor environment."""
    last_frame: Optional[np.ndarray] = None
    if not render_image:
        last_frame = controller.last_event.current_frame

    action_dict = {}
    if simplify_physics:
        action_dict["simplifyPhysics"] = True

    if action == PICKUP:
        action_dict["action"] = "PickupObject"

    elif action == DROP:
        action_dict["action"] = "ReleaseObject"

    elif action in [MOVE_AHEAD, MOVE_BACK, ROTATE_LEFT, ROTATE_RIGHT]:
        action_dict = {**action_dict, **ADDITIONAL_ARM_ARGS}
        if action == MOVE_AHEAD:
            action_dict["action"] = "MoveAhead"
            action_dict["moveMagnitude"] = move_agent_dist

        elif action == MOVE_BACK:
            action_dict["action"] = "MoveBack"
            action_dict["moveMagnitude"] = move_agent_dist

        elif action == ROTATE_RIGHT:
            action_dict["action"] = "RotateRight"
            action_dict["degrees"] = rotate_degrees

        elif action == ROTATE_LEFT:
            action_dict["action"] = "RotateLeft"
            action_dict["degrees"] = rotate_degrees

    elif "MoveArm" in action:
        action_dict = {**action_dict, **ADDITIONAL_ARM_ARGS}
        if "MoveArmHeight" in action:
            action_dict["action"] = "MoveArmBaseUp"

            if action == "MoveArmHeightP":
                action_dict["distance"] = move_base_dist
            if action == "MoveArmHeightM":
                action_dict["distance"] = -move_base_dist
        else:
            action_dict["action"] = "MoveArmRelative"
            offset = {"x": 0, "y": 0, "z": 0}

            axis, plus_or_minus = action[-2:].lower()
            assert axis in ["x", "y", "z"] and plus_or_minus in ["p", "m"]

            offset[axis] = (1 if plus_or_minus == "p" else -1) * move_arm_dist

            action_dict["offset"] = offset

    elif action.startswith("RotateArmWrist"):
        copy_additions = copy.deepcopy(ADDITIONAL_ARM_ARGS)
        action_dict = {**action_dict, **copy_additions}

        action_dict["action"] = "RotateWristRelative"

        tmp = action.replace("RotateArmWrist", "").lower()
        axis, plus_or_minus = tmp[:-1], tmp[-1]

        assert axis in ["pitch", "yaw", "roll"], plus_or_minus in ["p", "m"]

        action_dict[axis] = (
            1 if plus_or_minus == "p" else -1
        ) * rotate_wrist_degrees

    elif action.startswith("RotateArmElbow"):
        copy_additions = copy.deepcopy(ADDITIONAL_ARM_ARGS)
        action_dict = {**action_dict, **copy_additions}

        action_dict["action"] = "RotateElbowRelative"

        plus_or_minus = action[-1].lower()

        assert plus_or_minus in ["p", "m"]

        action_dict["degrees"] = (
            1 if plus_or_minus == "p" else -1
        ) * rotate_elbow_degrees

    elif action in [LOOK_UP, LOOK_DOWN]:
        copy_additions = copy.deepcopy(ADDITIONAL_ARM_ARGS)
        action_dict = {**action_dict, **copy_additions}
        if action == LOOK_UP:
            action_dict["action"] = LOOK_UP
        elif action == LOOK_DOWN:
            action_dict["action"] = LOOK_DOWN
    else:
        raise NotImplementedError

    sr = controller.step(action_dict)

    if not render_image:
        assert last_frame is not None
        controller.last_event.frame = last_frame

    return sr


@attr.s(kw_only=True)
class HouseAugmenter:
    p_randomize_materials: Optional[float] = attr.ib(
        validator=is_probability_validator, default=0.0
    )

    def apply(
        self,
        controller: Controller,
        random: Optional[Union[random.Random, ModuleType]] = random,
    ):
        if random.random() < self.p_randomize_materials:
            controller.step(
                action="RandomizeMaterials", raise_for_failure=True
            )
        else:
            controller.step(action="ResetMaterials", raise_for_failure=True)
        controller.step(action="randomizeObjectMass", raise_for_failure=True)


class ProcTHORDataset:
    def __init__(self, houses: Sequence[Dict[str, Any]]):
        self.houses = houses

        self.reachable_positions_map: Dict[int, Sequence[Vector3]] = {}

        self.relationships_graph_map: Dict[int, StrictMultiDiGraph] = {}
        self.soft_relationships_graph_map: Dict[int, StrictMultiDiGraph] = {}

        assert len(self) > 0

    def __len__(self):
        return len(self.houses)

    def initialize_house(
        self, controller: Controller, index: int
    ) -> Optional[Dict[str, Any]]:
        controller.reset("Procedural")

        house = self.houses[index]

        controller.step(
            action="CreateHouse",
            house=HouseUpgradeManager.upgrade_to(house, "1.0.0"),
            raise_for_failure=True,
        )

        if index not in self.reachable_positions_map:
            pose = house["metadata"]["agent"].copy()

            if controller.initialization_parameters["agentMode"] == "locobot":
                del pose["standing"]

            rot_offs = [i * 90 for i in range(4)]
            xoffs = [0.1 * i for i in [0, -1, 1]]
            zoffs = xoffs

            def add_to_vec3(
                vec3: Dict[str, float],
                x: float = 0,
                y: float = 0,
                z: float = 0,
            ):
                return {
                    "x": vec3["x"] + x,
                    "y": vec3["y"] + y,
                    "z": vec3["z"] + z,
                }

            assert "x" not in pose
            md: Optional[Dict[str, Any]] = None
            _pose = pose
            for rot_off, xoff, zoff in itertools.product(
                rot_offs, xoffs, zoffs
            ):
                _pose = {
                    **pose,
                    "position": add_to_vec3(pose["position"], x=xoff, z=zoff),
                    "rotation": add_to_vec3(pose["rotation"], y=rot_off),
                }
                md = controller.step(action="TeleportFull", **_pose).metadata
                if md["lastActionSuccess"]:
                    break

            house["metadata"]["agent"] = _pose

            if not md["lastActionSuccess"]:
                warnings.warn(f"Initial teleport failing in {index}.")
                return None

            rp_md = controller.step(action="GetReachablePositions").metadata
            if not rp_md["lastActionSuccess"]:
                warnings.warn(f"GetReachablePositions failed in {index}")
                return None

            reachable_positions = rp_md["actionReturn"]
            self.reachable_positions_map[index] = reachable_positions

            self.soft_relationships_graph_map[
                index
            ] = build_soft_relationship_graph(
                controller=controller, rooms=house["rooms"]
            )
            self.relationships_graph_map[index] = build_relationship_graph(
                controller=controller, rooms=house["rooms"]
            )

        return house


class BoolSetTask(Task):
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
        **kwargs,
    ):
        super().__init__(**prepare_locals_for_super(locals()))
        self.entity_0, self.relationship_type, self.entity_1 = task_info[
            "relationship"
        ]
        self.house = house

        self.took_done_action = False

        self.start_energy = min(self.compute_relationship_energies())
        self.last_energy = self.start_energy

    def compute_relationship_energies(self):
        soft_graph = build_soft_relationship_graph(
            controller=self.env, rooms=self.house["rooms"]
        )

        def entity_to_type(entity: str):
            if entity == "agent":
                return NodeTypeEnum.AGENT
            elif entity == "room":
                return NodeTypeEnum.ROOM
            else:
                return NodeTypeEnum.OBJECT

        def check_node_match(
            entity: str, entity_type: NodeTypeEnum, node_data: Dict[str, Any]
        ):
            if node_data["type"] != entity_type:
                return False
            if entity_type == NodeTypeEnum.AGENT:
                return True
            elif entity_type == NodeTypeEnum.OBJECT:
                return entity == node_data["object_type"]
            elif entity_type == NodeTypeEnum.ROOM:
                return entity == node_data["room_type"]
            else:
                raise NotImplementedError

        entity_0_type = entity_to_type(self.entity_0)
        entity_1_type = entity_to_type(self.entity_1)

        energies = []
        for node, node_data in soft_graph.nodes(data=True):
            if check_node_match(
                entity=self.entity_0,
                entity_type=entity_0_type,
                node_data=node_data,
            ):
                for e in soft_graph.edges(node, data=True, keys=True):
                    _, other_node, key, edge_data = e

                    if key == self.relationship_type and check_node_match(
                        entity=self.entity_1,
                        entity_type=entity_1_type,
                        node_data=soft_graph.nodes[other_node],
                    ):
                        energies.append(edge_data["weight"])

        return energies

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(len(self.ACTIONS))

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def _step(self, action: int) -> RLStepResult:
        action = self.ACTIONS[action]
        if action == DONE:
            self.took_done_action = True
        else:
            arm_agent_step(controller=self.env, action=action)

        energy = min(self.compute_relationship_energies())

        reward = 10 * (self.last_energy - energy)
        self.last_energy = energy

        reward += 10 * (action == DONE) * (energy == 0.0)

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
            **super(BoolSetTask, self).metrics(),
            "energy_at_start": self.start_energy,
            "energy_remaining": self.last_energy,
            "success": self.last_energy == 0.0,
        }

    def reached_terminal_state(self) -> bool:
        return self.took_done_action

    def close(self) -> None:
        self.env.stop()


class BoolSetTaskSampler(TaskSampler):
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
            branch="nanna-grasp-force",
        )
        self.controller_kwargs.update(controller_kwargs)

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

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[BoolSetTask]:
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

        # Randomize agent pose
        n_retries = 10
        for i in range(n_retries):
            standing = (
                {}
                if self.controller.initialization_parameters["agentMode"]
                == "locobot"
                else {"standing": self._random.choice([False, True])}
            )
            starting_pose = AgentPose(
                position=self._random.choice(
                    self.procthor_dataset.reachable_positions_map[
                        self._current_house_ind
                    ]
                ),
                rotation=Vector3(x=0, y=self._random.random() * 360, z=0),
                horizon=self._random.randint(-1, 2) * 30,
                **standing,
            )
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
        self.augmenter.apply(self.controller)

        # Pick relationship via rejection sampling
        soft_relationships_graph = (
            self.procthor_dataset.soft_relationships_graph_map[
                self._current_house_ind
            ]
        )
        object_nodes = []
        object_types = set()
        room_nodes = []
        agent_node = None
        for node, node_data in soft_relationships_graph.nodes(data=True):
            if node_data["type"] == NodeTypeEnum.OBJECT:
                object_nodes.append((node, node_data))
                object_types.add(node_data["object_type"])
            elif node_data["type"] == NodeTypeEnum.ROOM:
                room_nodes.append((node, node_data))
            elif node_data["type"] == NodeTypeEnum.AGENT:
                assert agent_node is None
                agent_node = (node, node_data)
            else:
                raise NotImplementedError
        object_types = list(object_types)

        objects = self.controller.last_event.metadata["objects"]
        object_type_to_objects = defaultdict(lambda: [])
        for o in objects:
            object_type_to_objects[o["objectType"]].append(o)

        oids_for_object_filter = []
        while True:
            relationship_pair = self._random.choice(
                # list(EntityPairEnum)
                [EntityPairEnum.AGENT_OBJ]
            )

            if relationship_pair == EntityPairEnum.AGENT_OBJ:
                ot = self._random.choice(object_types)
                if len(object_type_to_objects[ot]) == 0:
                    warnings.warn(
                        f"Cannot find object of type: {ot} in house {self._current_house_ind}."
                    )
                    continue
                o_instance = object_type_to_objects[ot][0]

                possible_relationships = [
                    RelationshipEnum.SEES,
                    RelationshipEnum.TOUCHES,
                ]
                if o_instance["pickupable"]:
                    possible_relationships.append(RelationshipEnum.HOLDS)

                relationship_type = self._random.choice(possible_relationships)

                relationship = ("agent", relationship_type, ot)

                oids_for_object_filter.extend(
                    [o["objectId"] for o in object_type_to_objects[ot]]
                )
                break
            else:
                raise NotImplementedError

        self.controller.step(
            action="SetObjectFilter",
            objectIds=oids_for_object_filter,
            raise_for_failure=True,
        )

        self._last_sampled_task = BoolSetTask(
            env=self.controller,
            sensors=self.sensors,
            max_steps=self.max_steps,
            house=self._current_house,
            task_info={
                "relationship": relationship,
                "house_md5": md5_hash_str_as_int(
                    canonicaljson.encode_canonical_json(
                        self._current_house
                    ).decode("utf-8")
                ),
            },
        )
        return self._last_sampled_task
