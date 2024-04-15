import os
import traceback
from typing import Dict, List, Union

import compress_pickle
import h5py
import numpy as np
import ray
import torch
import tqdm
from allenact.utils.inference import InferenceAgent
from allenact.utils.system import get_logger
from procthor_baseline_configs.one_phase.one_phase_rgb_clip_dagger_multi_node import (
    ProcThorOnePhaseRGBClipResNet50DaggerTrainMultiNodeConfig as Config,
)
from rearrange.environment import RearrangeTaskSpec


def task_spec_bytes_to_hdf5_path(rts: Union[bytes, RearrangeTaskSpec]):
    if not isinstance(rts, RearrangeTaskSpec):
        rts = RearrangeTaskSpec(
            **compress_pickle.loads(rts, compression="gzip")
        )
    return os.path.join(
        os.environ["TRAJECTORY_DIR_PATH"],
        "data",
        "trajectories_with_memory",
        f"{rts.scene}",
        f"{rts.unique_id}.hdf5",
    )


@ray.remote(num_cpus=2)  # type: ignore
@torch.no_grad()
def rollout_agent(
    worker_ind: int,
    sampler_args: Dict,
):
    config = Config()
    os.environ["TRAJECTORY_DIR_PATH"]
    devices = (
        [torch.device("cpu")]
        if not torch.cuda.is_available()
        else list(range(torch.cuda.device_count()))  # type: ignore
    )
    agent = InferenceAgent.from_experiment_config(
        exp_config=config,
        checkpoint_path=os.environ["CKPT_PATH"],
        device=devices[worker_ind % len(devices)],
    )
    task_sampler = config.make_sampler_fn(**sampler_args)
    task_sampler.task_spec_iterator.scenes_to_task_spec_dicts = {
        scene: [
            b
            for b in bytes_list
            if not (lambda p: os.path.exists(p))(
                task_spec_bytes_to_hdf5_path(b)
            )
        ]
        for scene, bytes_list in task_sampler.task_spec_iterator.scenes_to_task_spec_dicts.items()
    }
    for (scene, bytes_list,) in list(
        task_sampler.task_spec_iterator.scenes_to_task_spec_dicts.items()
    ):
        if len(bytes_list) == 0:
            get_logger().info(
                f"Worker {worker_ind}: {scene} already complete."
            )
            del task_sampler.task_spec_iterator.scenes_to_task_spec_dicts[
                scene
            ]

    scene = ""
    try:
        task = task_sampler.next_task()
    except:
        task = None

    while task is not None:
        if scene != "" and scene != task_sampler.current_task_spec.scene:
            get_logger().info(f"Worker {worker_ind}: Finished scene {scene}")

        scene = task_sampler.current_task_spec.scene
        get_logger().info(
            f"Worker {worker_ind} starting {task_sampler.current_task_spec.unique_id}"
        )

        # prepare trajectory dump path
        hdf5_path = task_spec_bytes_to_hdf5_path(
            task_sampler.current_task_spec
        )
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
        assert not os.path.exists(hdf5_path)

        agent.reset()
        observations = task.get_observations()
        visible_objects_list: List[List[str]] = []
        agent_pos_list: List[Dict] = []

        memory_list = []
        while not task.is_done():
            action = agent.act(observations=observations)
            visible_objects = [
                o["objectId"]
                for o in task.env.controller.last_event.metadata["objects"]
                if o["visible"]
            ]
            visible_objects_list.append(visible_objects)
            agent_pos_list.append(
                task.env.controller.last_event.metadata["agent"]
            )
            observations = task.step(action).observation
            memory = agent.memory.tensor("rnn")
            memory_list.append(memory.detach().cpu().numpy())

        try:
            before_update_info = dict(
                next_value=0,
                use_gae=True,
                gamma=0.99,
                tau=0.95,
                adv_stats_callback=lambda advantages: {
                    "mean": advantages.mean(),
                    "std": advantages.std(),
                },
            )
            agent.rollout_storage.before_updates(**before_update_info)
            batch = next(
                agent.rollout_storage.batched_experience_generator(
                    num_mini_batch=1
                )
            )
            metrics = task.metrics()

            with h5py.File(hdf5_path, "w") as hdf5_file:
                obs_group = hdf5_file.create_group("observations")
                for obs_key in batch["observations"]:
                    obs_group.create_dataset(
                        obs_key,
                        data=batch["observations"][obs_key]
                        .type(torch.float16)
                        .cpu()
                        .numpy(),
                        compression="gzip",
                        compression_opts=6,
                    )

                mem_group = hdf5_file.create_group("memory")
                for mem_key in batch["memory"]:
                    mem_group.create_dataset(
                        mem_key,
                        data=np.stack(memory_list, axis=0),
                        compression="gzip",
                        compression_opts=6,
                    )

                hdf5_file.create_dataset(
                    "actions",
                    data=batch["actions"].cpu().numpy(),
                    compression="gzip",
                    compression_opts=6,
                )
                for metric_key in [
                    "success",
                    "prop_fixed_strict",
                    "num_initially_misplaced",
                ]:
                    hdf5_file.create_dataset(
                        metric_key,
                        data=np.array([metrics[f"unshuffle/{metric_key}"]]),
                    )

            torch.save(
                (visible_objects_list, agent_pos_list),
                hdf5_path.replace(".hdf5", ".pt"),
            )

            get_logger().info(
                f"Worker {worker_ind}: Completed {task_sampler.current_task_spec.unique_id}"
            )
        except:
            get_logger().error(
                f"Worker {worker_ind}: Failed to save {hdf5_path}. Traceback:\n {traceback.format_exc()}"
            )

            if os.path.exists(hdf5_path):
                os.remove(hdf5_path)

        try:
            task = task_sampler.next_task()
        except:
            task = None

    task_sampler.close()


def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


def main():
    ray.init()
    assert "TRAJECTORY_DIR_PATH" in os.environ, "Must set TRAJECTORY_DIR_PATH"
    assert (
        "CKPT_PATH" in os.environ
    ), "CKPT_PATH must be set in the environment"
    assert (
        "NPIECES" in os.environ
    ), "NPIECES must be set in the environment"  # try 120
    config = Config()
    devices = (
        [torch.device("cpu")]
        if not torch.cuda.is_available()
        else list(range(torch.cuda.device_count()))
    )
    job_list = [
        rollout_agent.remote(
            i,
            dict(
                config.train_task_sampler_args(
                    process_ind=i,
                    total_processes=int(os.environ["NPIECES"]),
                    devices=devices,
                ),
                epochs=1,
            ),
        )
        for i in range(int(os.environ["NPIECES"]))
    ]
    for i in tqdm.tqdm(to_iterator(job_list), total=len(job_list)):
        pass


if __name__ == "__main__":
    main()
