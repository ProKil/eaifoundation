# Git version of procthor_rearrangement: 549af93a5e1c4aee8d77b613b0a73e3fc586cf61

import os

import torch
from allenact.utils.inference import InferenceAgent
from procthor_baseline_configs.one_phase.one_phase_rgb_clip_dagger_multi_node import (
    ProcThorOnePhaseRGBClipResNet50DaggerTrainMultiNodeConfig as Config,
)


@torch.no_grad()
def main():
    assert (
        "CKPT_PATH" in os.environ
    ), "CKPT_PATH must be set in the environment"
    config = Config()
    agent = InferenceAgent.from_experiment_config(
        exp_config=config,
        checkpoint_path=os.environ["CKPT_PATH"],
        device=torch.device("cuda:0"),
    )

    task_sampler = config.make_sampler_fn(
        **config.train_task_sampler_args(process_ind=0, total_processes=1)
    )

    for ind in range(1000):
        agent.reset()

        task = task_sampler.next_task()
        print(task_sampler.current_task_spec.scene)
        observations = task.get_observations()

        cumulative_object_set = set()
        while not task.is_done():
            action = agent.act(observations=observations)
            cumulative_object_set |= set(
                o["name"]
                for o in task.env.controller.last_event.metadata["objects"]
                if o["visible"]
            )
            print(len(cumulative_object_set))
            observations = task.step(action).observation


if __name__ == "__main__":
    main()
