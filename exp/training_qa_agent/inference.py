# Git version of procthor_rearrangement: 549af93a5e1c4aee8d77b613b0a73e3fc586cf61

import os
from typing import Any, Dict

import torch
from allenact.utils.inference import InferenceAgent

from exp.training_qa_agent.qa_experiments import QAConfig as Config

# def traverse_tree_and_expand_(dictionary: Dict[str, Any], scale=1):
#     for _, value in dictionary.items():
#         if isinstance(value, torch.Tensor):
#             value = value.expand(scale, *value.shape[1:])
#         elif isinstance(value, dict):
#             traverse_tree_and_expand_(value, scale)


# @torch.no_grad()
def main():
    config = Config()
    agent = InferenceAgent.from_experiment_config(
        exp_config=config,
        device=torch.device("cuda:0"),
    )

    task_sampler = config.make_sampler_fn(
        **config.train_task_sampler_args(
            process_ind=0, total_processes=1, devices=[torch.device("cuda:0")]
        )
    )

    for ind in range(1):
        agent.reset()
        agent.actor_critic.train()

        task = task_sampler.next_task()
        observations = task.get_observations()

        cumulative_object_set = set()
        while not task.is_done():
            action = agent.act(observations=observations)
            loss = agent.memory["working_memory"][0].sum()
            loss.backward()
            observations = task.step(action).observation


if __name__ == "__main__":
    main()
