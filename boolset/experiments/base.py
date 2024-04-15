import abc
import platform
import warnings
from abc import ABC
from math import ceil
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import ai2thor
import ai2thor.build
import ai2thor.fifo_server
import ai2thor.platform
import datasets
import gym
import numpy as np
import torch
from allenact.base_abstractions.experiment_config import (
    ExperimentConfig,
    MachineParams,
)
from allenact.base_abstractions.preprocessor import (
    SensorPreprocessorGraph,
)
from allenact.base_abstractions.sensor import (
    ExpertActionSensor,
    SensorSuite,
)
from allenact.base_abstractions.task import TaskSampler
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from allenact.utils.experiment_utils import (
    evenly_distribute_count_into_bins,
)
from allenact_plugins.ithor_plugin.ithor_util import (
    get_open_x_displays,
    horizontal_to_vertical_fov,
)
from torch.distributions.utils import lazy_property

import prior
from boolset.tasks_and_samplers import (
    BoolSetTask,
    BoolSetTaskSampler,
    HouseAugmenter,
    ProcTHORDataset,
)


class BaseConfig(ExperimentConfig, ABC):
    STEP_SIZE = 0.2
    ROTATION_DEGREES = 45.0
    VISIBILITY_DISTANCE = 1.5
    STOCHASTIC = False
    HORIZONTAL_FIELD_OF_VIEW = 100

    CAMERA_WIDTH = 224
    CAMERA_HEIGHT = 224
    SCREEN_SIZE = 224
    MAX_STEPS = 500

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None

    DEFAULT_NUM_TRAIN_PROCESSES: Optional[int] = None
    DEFAULT_TRAIN_GPU_IDS = tuple(range(torch.cuda.device_count()))
    DEFAULT_VALID_GPU_IDS = (torch.cuda.device_count() - 1,)
    DEFAULT_TEST_GPU_IDS = (torch.cuda.device_count() - 1,)

    DEFAULT_USE_WEB_RENDER = False

    THOR_COMMIT_ID: Optional[str] = "82097eaa409744cd0243ccc01e79507171eb3a72"

    ACTION_SPACE = gym.spaces.Discrete(len(BoolSetTask.ACTIONS))

    def __init__(
        self,
        num_train_processes: Optional[int] = None,
        num_test_processes: Optional[int] = None,
        test_on_validation: bool = False,
        train_gpu_ids: Optional[Sequence[int]] = None,
        val_gpu_ids: Optional[Sequence[int]] = None,
        test_gpu_ids: Optional[Sequence[int]] = None,
        use_web_render: Optional[bool] = None,
    ):
        super().__init__()

        def v_or_default(v, default):
            return v if v is not None else default

        self.num_train_processes = v_or_default(
            num_train_processes, self.DEFAULT_NUM_TRAIN_PROCESSES
        )
        self.num_test_processes = v_or_default(
            num_test_processes, (10 if torch.cuda.is_available() else 1)
        )
        self.test_on_validation = test_on_validation
        self.train_gpu_ids = v_or_default(
            train_gpu_ids, self.DEFAULT_TRAIN_GPU_IDS
        )
        self.val_gpu_ids = v_or_default(
            val_gpu_ids, self.DEFAULT_VALID_GPU_IDS
        )
        self.test_gpu_ids = v_or_default(
            test_gpu_ids, self.DEFAULT_TEST_GPU_IDS
        )

        self.use_web_render = v_or_default(
            use_web_render, self.DEFAULT_USE_WEB_RENDER
        )

        self.sampler_devices = self.train_gpu_ids

        self.extra_losses = None

        assert (
            len(self.split_to_procthor_houses["train"]) > 0
        )  # Forces download

    @abc.abstractmethod
    def sensors(self):
        raise NotImplementedError

    @abc.abstractmethod
    def preprocessors(self):
        raise NotImplementedError

    @lazy_property
    def split_to_procthor_houses(self) -> Dict[str, datasets.Dataset]:
        debug_config = {
            "train_size": 10,
            "val_size": 1,
            "test_size": 1,
        }
        return prior.load_dataset(
            "procthor-10k",
            revision="size-control",
            # config=None if torch.cuda.is_available() else debug_config,
        )

    def controller_kwargs(self):
        assert self.THOR_COMMIT_ID is not None
        assert not self.STOCHASTIC

        return dict(
            commit_id=self.THOR_COMMIT_ID,
            server_class=ai2thor.fifo_server.FifoServer,
            include_private_scenes=False,
            fastActionEmit=True,
            snapToGrid=False,
            autoSimulation=False,
            autoSyncTransforms=True,
            width=self.CAMERA_WIDTH,
            height=self.CAMERA_HEIGHT,
            fieldOfView=horizontal_to_vertical_fov(
                horizontal_fov_in_degrees=self.HORIZONTAL_FIELD_OF_VIEW,
                width=self.CAMERA_WIDTH,
                height=self.CAMERA_HEIGHT,
            ),
            makeAgentsVisible=True,
            renderDepthImage=any(
                isinstance(s, DepthSensor) for s in self.sensors()
            ),
            visibilityScheme="Distance",
            agentMode="arm",
            rotateStepDegrees=self.ROTATION_DEGREES,
            visibilityDistance=self.VISIBILITY_DISTANCE,
            gridSize=self.STEP_SIZE,
            useMassThreshold=True,
            massThreshold=10,
            platform="CloudRendering",
        )

    def machine_params(self, mode="train", **kwargs):
        sampler_devices: Sequence[torch.device] = []
        devices: Sequence[torch.device]
        if mode == "train":
            workers_per_device = 1
            devices = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else cast(Tuple, self.train_gpu_ids) * workers_per_device
            )
            nprocesses = evenly_distribute_count_into_bins(
                self.num_train_processes, max(len(devices), 1)
            )
            sampler_devices = self.sampler_devices
        elif mode == "valid":
            nprocesses = 1
            devices = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else self.val_gpu_ids
            )
        elif mode == "test":
            devices = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else self.test_gpu_ids
            )
            nprocesses = evenly_distribute_count_into_bins(
                self.num_test_processes, max(len(devices), 1)
            )
        else:
            raise NotImplementedError(
                "mode must be 'train', 'valid', or 'test'."
            )

        sensors = [*self.sensors()]
        if mode != "train":
            sensors = [
                s for s in sensors if not isinstance(s, ExpertActionSensor)
            ]

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(
                    sensors
                ).observation_spaces,
                preprocessors=self.preprocessors(),
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=sampler_devices
            if mode == "train"
            else devices,  # ignored with > 1 gpu_ids
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return BoolSetTaskSampler(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(
            np.linspace(0, n, num_parts + 1, endpoint=True)
        ).astype(np.int32)

    def _get_sampler_args_for_scene_split(
        self,
        split: str,
        houses: Sequence[Dict[str, Any]],
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]],
        deterministic_cudnn: bool,
        include_expert_sensor: bool = True,
        allow_oversample: bool = False,
    ) -> Dict[str, Any]:

        oversample_warning = (
            f"Warning: oversampling some of the houses (of {len(houses)}) to feed all processes ({total_processes})."
            " You can avoid this by setting a number of workers divisible by the number of scenes"
        )
        if total_processes > len(houses):  # oversample some scenes -> bias
            if not allow_oversample:
                raise RuntimeError(
                    f"Cannot have `total_processes > len(scenes)`"
                    f" ({total_processes} > {len(houses)}) when `allow_oversample` is `False`."
                )

            if total_processes % len(houses) != 0:
                warnings.warn(oversample_warning)
            houses = list(houses) * int(ceil(total_processes / len(houses)))
            houses = houses[
                : total_processes * (len(houses) // total_processes)
            ]
        elif len(houses) % total_processes != 0:
            warnings.warn(oversample_warning)

        inds = self._partition_inds(len(houses), total_processes)

        if self.use_web_render:
            device_dict = dict(
                gpu_device=devices[process_ind % len(devices)],
                platform=ai2thor.platform.CloudRendering,
            )
        else:
            x_display: Optional[str] = None
            if platform.system() == "Linux":
                x_displays = get_open_x_displays(throw_error_if_empty=True)

                if len([d for d in devices if d != torch.device("cpu")]) > len(
                    x_displays
                ):
                    warnings.warn(
                        f"More GPU devices found than X-displays (devices: `{x_displays}`, x_displays: `{x_displays}`)."
                        f" This is not necessarily a bad thing but may mean that you're not using GPU memory as"
                        f" efficiently as possible. Consider following the instructions here:"
                        f" https://allenact.org/installation/installation-framework/#installation-of-ithor-ithor-plugin"
                        f" describing how to start an X-display on every GPU."
                    )
                x_display = x_displays[process_ind % len(x_displays)]

            device_dict = dict(x_display=x_display)

        training = split == "train"
        return {
            "procthor_dataset": ProcTHORDataset(
                [
                    houses[i]
                    for i in range(inds[process_ind], inds[process_ind + 1])
                ]
            ),
            "house_repeats": float("inf") if training else 1,
            "reset_on_scene_replay": not training,
            "augmenter": HouseAugmenter(p_randomize_materials=0.8)
            if training
            else None,
            "max_steps": self.MAX_STEPS,
            "sensors": [
                s
                for s in self.sensors()
                if (
                    include_expert_sensor
                    or not isinstance(s, ExpertActionSensor)
                )
            ],
            "seed": seeds[process_ind] if seeds is not None else None,
            "controller_kwargs": {**self.controller_kwargs(), **device_dict},
        }

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            split="train",
            houses=self.split_to_procthor_houses["train"],
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            allow_oversample=True,
        )
        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            split="valid",
            houses=self.split_to_procthor_houses["valid"],
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            allow_oversample=True,
        )
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:

        if self.test_on_validation:
            if not self.test_on_validation:
                warnings.warn(
                    "`test_on_validation` is set to `True` and thus we will run evaluation on the validation set instead."
                    " Be careful as the saved metrics json and tensorboard files **will still be labeled as"
                    " 'test' rather than 'valid'**."
                )
            else:
                warnings.warn(
                    "No test dataset dir detected, running test on validation set instead."
                    " Be careful as the saved metrics json and tensorboard files *will still be labeled as"
                    " 'test' rather than 'valid'**."
                )

            return self.valid_task_sampler_args(
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            )

        else:
            res = self._get_sampler_args_for_scene_split(
                split="test",
                houses=self.split_to_procthor_houses["test"],
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
                allow_oversample=True,
            )
            return res
