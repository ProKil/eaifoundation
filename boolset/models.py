from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import gym
import gym.spaces
import torch
import torch.nn as nn
from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    DistributionType,
    LinearActorHead,
    LinearCriticHead,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import (
    ActorCriticOutput,
    Memory,
    ObservationType,
)
from allenact.embodiedai.models.aux_models import AuxiliaryModel
from allenact.utils.model_utils import FeatureEmbedding
from clip.model import LayerNorm, Transformer
from gym.spaces.dict import Dict as SpaceDict


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class RecurrentTransformer(nn.Module):
    # Modified from the VisionTransformer from https://github.com/openai/CLIP/blob/b46f5ac7587d2e1862f8b7b1573179d80dcdd620/clip/model.py

    def __init__(
        self,
        num_inputs: int,
        num_memory: int,
        width: int,
        layers: int,
        heads: int,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_memory = num_memory
        self.width = width
        self.layers = layers
        self.heads = heads

        scale = width**-0.5
        # self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(num_inputs, width)
        )
        self.initial_memory_embedding = nn.Parameter(
            scale * torch.randn(num_memory, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width=width, layers=layers, heads=heads)

        # self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, width))

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_reset_mask: torch.Tensor,
    ):
        num_steps, num_samplers, num_inputs, width = x.shape

        assert num_inputs == self.num_inputs and width == self.width
        assert memory.shape == (1, num_samplers, self.num_memory, width)
        assert memory_reset_mask.shape == (num_steps, num_samplers, 1)

        memory_reset_mask = memory_reset_mask.unsqueeze(-1)
        memory = memory[0]  # shape == (nsamplers, num_memory, width)

        x = x + self.positional_embedding.unsqueeze(0).unsqueeze(0).to(x.dtype)

        x_for_step_embs = []
        memory_for_step_embs = []
        for step in range(num_steps):
            x_for_step = x[step]  # shape == (nsamplers, ninputs, width)
            memory_reset_mask_for_step = memory_reset_mask[
                step
            ]  # shape == (nsamplers, 1, 1)
            memory = (memory * memory_reset_mask_for_step) + (
                1 - memory_reset_mask_for_step
            ) * self.initial_memory_embedding.unsqueeze(0)

            y = torch.cat((x_for_step, memory), dim=1)
            y = self.ln_pre(y)

            y = y.permute(1, 0, 2)  # NLD -> LND
            y = self.transformer(y)
            y = y.permute(1, 0, 2)  # LND -> NLD
            # y = self.ln_post(y)

            x_for_step_emb = y[:, : -self.num_memory]
            memory = y[:, -self.num_memory :]

            x_for_step_embs.append(x_for_step_emb)
            memory_for_step_embs.append(memory)

        # if self.proj is not None:
        #     x = x @ self.proj

        return torch.stack(memory_for_step_embs, dim=0), memory.unsqueeze(0)


class RelationshipTransformer(nn.Module):
    # Modified from the VisionTransformer from https://github.com/openai/CLIP/blob/b46f5ac7587d2e1862f8b7b1573179d80dcdd620/clip/model.py

    def __init__(
        self,
        num_entities: int,
        num_relationship_types: int,
        max_entity_instances: int,
        max_input_relationships: int,
        num_memory: int,
        width: int,
        layers: int,
        heads: int,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relationship_types = num_relationship_types
        self.max_entity_instances = max_entity_instances
        self.max_input_relationships = max_input_relationships

        self.num_memory = num_memory
        self.width = width
        self.layers = layers
        self.heads = heads

        scale = width**-0.5
        self.positional_embedding_mat = nn.Parameter(
            scale * torch.randn(self.max_input_relationships * 3, width)
        )

        def create_embedding(num_embeddings: int):
            emb = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=width
            )
            emb.weight.data.normal_(std=scale)
            return emb

        self.entity_embedding = create_embedding(self.num_entities)
        self.relationship_type_embedding = create_embedding(
            self.num_relationship_types
        )
        self.instance_embedding = create_embedding(self.max_entity_instances)

        self.null_embedding = nn.Parameter(scale * torch.randn(1, width))
        self.class_embedding = nn.Parameter(scale * torch.randn(1, width))

        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width=width, layers=layers, heads=heads)

        self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, width))

    def forward(
        self,
        e0: torch.Tensor,
        i0: torch.Tensor,
        e1: torch.Tensor,
        i1: torch.Tensor,
        relationship_type: torch.Tensor,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(e0.shape) == 3
        assert (
            e0.shape
            == i0.shape
            == e1.shape
            == i1.shape
            == relationship_type.shape
        )

        num_steps, num_samplers, num_relationships = e0.shape

        assert num_relationships == self.max_input_relationships
        assert memory.shape[:2] == (num_steps, num_samplers)

        rel_type_emb = self.relationship_type_embedding(
            relationship_type.long()
        )
        e0_emb = self.entity_embedding(e0.long()) + self.instance_embedding(
            i0.long()
        )
        e1_emb = self.entity_embedding(e1.long()) + self.instance_embedding(
            i1.long()
        )

        x = torch.cat(
            (rel_type_emb, e0_emb, e1_emb), dim=2
        ) + self.positional_embedding_mat.unsqueeze(0).unsqueeze(0)

        cls_init_emb = self.class_embedding.view(1, 1, 1, -1).repeat(
            num_steps, num_samplers, 1, 1
        )
        x = torch.cat(
            (
                cls_init_emb,
                x,
                memory,
            ),
            dim=2,
        )

        x_shape_full = x.shape

        x = x.view(num_steps * num_samplers, *x.shape[-2:])

        x_out = self.ln_pre(x)
        x_out = x_out.permute(1, 0, 2)
        x_out = self.transformer(x_out)
        x_out = x_out.permute(1, 0, 2)
        x_out = self.ln_post(x_out)
        x_out = x_out.view(*x_shape_full)

        cls_emb = x_out[:, :, 0]
        rel_type_emb = x_out[:, :, 1 : (num_relationships + 1)]

        return cls_emb, rel_type_emb


class Unsqueeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(self.dim)


class Permute(nn.Module):
    def __init__(self, permutation: Tuple[int, ...]):
        super().__init__()
        self.permutation = permutation

    def forward(self, x: torch.Tensor):
        return x.permute(self.permutation)


class FlattenNext(nn.Module):
    def __init__(self, start_dim: int):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: torch.Tensor):
        shape = list(x.shape)
        shape = (
            *shape[: self.start_dim],
            shape[self.start_dim] * shape[self.start_dim + 1],
            *shape[self.start_dim + 2 :],
        )
        return x.view(shape)


class RecurrentRelationshipTransformerActorCritic(
    ActorCriticModel[CategoricalDistr]
):
    action_space: gym.spaces.Discrete

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        obs_sensor_uuids: Sequence[str],
        relationship_sensor_uuid: str,
        obs_transformer_layers: int = 1,
        rel_transformer_layers: int = 1,
        memory_shape=(10, 768),
        auxiliary_uuids: Optional[List[str]] = None,
    ):
        super().__init__(
            action_space=action_space, observation_space=observation_space
        )
        self.memory_shape = memory_shape
        self.obs_sensor_uuids = obs_sensor_uuids
        self.relationship_sensor_uuid = relationship_sensor_uuid

        self.obs_transformer_layers = obs_transformer_layers
        self.rel_transformer_layers = rel_transformer_layers

        self.auxiliary_uuids = (
            auxiliary_uuids if auxiliary_uuids is not None else []
        )

        uuid_to_input_encoder = {}
        self.uuid_to_num_inputs = {}
        for uuid in obs_sensor_uuids:
            obs_space = observation_space[uuid]
            assert isinstance(
                obs_space, (gym.spaces.Box, gym.spaces.MultiDiscrete)
            )
            if len(obs_space.shape) == 1:
                # Assumes vector input
                uuid_to_input_encoder[uuid] = nn.Sequential(
                    Unsqueeze(2),
                    nn.Linear(
                        obs_space.shape[0], memory_shape[-1], bias=False
                    ),
                )
                self.uuid_to_num_inputs[uuid] = 1
            elif len(obs_space.shape) == 2:
                # Assumes features are along the -1 axis
                uuid_to_input_encoder[uuid] = nn.Linear(
                    obs_space.shape[-1], memory_shape[-1], bias=False
                )
                self.uuid_to_num_inputs[uuid] = obs_space.shape[0]
            elif len(obs_space.shape) == 3:
                # Assumes CNN shaped input so that channels are first
                uuid_to_input_encoder[uuid] = nn.Sequential(
                    FlattenNext(start_dim=3),
                    Permute((0, 1, 3, 2)),
                    nn.Linear(
                        obs_space.shape[0], memory_shape[-1], bias=False
                    ),
                )
                self.uuid_to_num_inputs[uuid] = (
                    obs_space.shape[1] * obs_space.shape[2]
                )
            else:
                raise NotImplementedError

        self.uuid_to_obs_sensor_encoder: Optional[
            nn.ModuleDict
        ] = nn.ModuleDict(uuid_to_input_encoder)

        self.memory_transformer = RecurrentTransformer(
            num_inputs=sum(self.uuid_to_num_inputs.values())
            + 1,  # + 1 for prev action
            num_memory=self.memory_shape[0],
            width=self.memory_shape[1],
            layers=self.obs_transformer_layers,
            heads=self.memory_shape[1] // 64,
        )

        assert all(
            isinstance(v, gym.spaces.MultiDiscrete)
            for v in self.observation_space[
                self.relationship_sensor_uuid
            ].spaces.values()
        )
        rel_sensor_spaces = self.observation_space[
            self.relationship_sensor_uuid
        ]
        self.relationship_transformer = RelationshipTransformer(
            num_entities=rel_sensor_spaces["e0"].nvec.max(),
            num_relationship_types=rel_sensor_spaces[
                "relationship_type"
            ].nvec.max(),
            max_entity_instances=len(rel_sensor_spaces["e0"].nvec),
            max_input_relationships=len(
                rel_sensor_spaces["relationship_type"].nvec
            ),
            num_memory=self.memory_shape[0],
            width=self.memory_shape[1],
            layers=self.rel_transformer_layers,
            heads=self.memory_shape[1] // 64,
        )

        self.aux_models: Optional[nn.ModuleDict] = None
        self.aux_fusion_logits: Optional[nn.ParameterDict] = None

        self.actor: Optional[LinearActorHead] = None
        self.critic: Optional[LinearCriticHead] = None

        self.create_actorcritic_head()
        self.create_aux_models()

        self.prev_action_embedder: FeatureEmbedding = FeatureEmbedding(
            self.action_space.n + 1, self.memory_shape[1]
        )

    def create_actorcritic_head(self):
        self.actor = LinearActorHead(
            self.memory_shape[-1], self.action_space.n
        )
        self.critic = LinearCriticHead(self.memory_shape[-1])

    def create_aux_models(self):
        if len(self.auxiliary_uuids) == 0:
            return

        aux_models = OrderedDict()
        fusion_logits = OrderedDict()
        for aux_uuid in self.auxiliary_uuids:
            aux_models[aux_uuid] = AuxiliaryModel(
                aux_uuid=aux_uuid,
                action_dim=self.action_space.n,
                obs_embed_dim=self.memory_shape[1],
                belief_dim=self.memory_shape[1],
                action_embed_size=self.memory_shape[1],
            )
            fusion_logits[aux_uuid] = nn.Parameter(
                torch.normal(mean=0.0, std=0.1, size=(self.memory_shape[0],))
            )

        self.aux_models = nn.ModuleDict(aux_models)
        self.aux_fusion_logits = nn.ParameterDict(fusion_logits)

    def _recurrent_memory_specification(self):
        return {
            "working_memory": (
                (
                    ("sampler", None),
                    ("memory_index", self.memory_shape[0]),
                    ("memory_emb", self.memory_shape[1]),
                ),
                torch.float32,
            )
        }

    def forward_encoder(self, observations: ObservationType) -> torch.Tensor:
        encodings = []
        for uuid in self.obs_sensor_uuids:
            encodings.append(
                self.uuid_to_obs_sensor_encoder[uuid](observations[uuid])
            )

        return torch.cat(encodings, dim=2)

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values.

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the working memory from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to memory states. See `RecurrentTransformer`.
        # Returns
        Tuple of the `ActorCriticOutput` new memory states.
        """

        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        obs_embeds = self.forward_encoder(observations)

        # 1.2 use embedding model to get prev_action embeddings
        assert self.prev_action_embedder.input_size == self.action_space.n + 1
        prev_actions_embeds = self.prev_action_embedder(
            torch.where(
                condition=0 != masks.view(*prev_actions.shape),
                input=prev_actions + 1,
                other=torch.zeros_like(prev_actions),
            )
        ).unsqueeze(2)

        joint_embeds = torch.cat(
            (obs_embeds, prev_actions_embeds), dim=2
        )  # (T, N, O, -1)

        # 2. use transformer to update working memory

        x, working_memory = self.memory_transformer(
            x=joint_embeds,
            memory=memory.tensor("working_memory").unsqueeze(0),
            memory_reset_mask=masks,
        )
        memory.set_tensor(
            "working_memory", working_memory.squeeze(0)
        )  # update memory here

        # 3. fuse memory for aux models
        aux_to_fused_memory = {}
        for aux_uuid in self.auxiliary_uuids:
            probs = torch.softmax(self.aux_fusion_logits[aux_uuid], dim=-1)
            aux_to_fused_memory[aux_uuid] = (x * probs.view(1, 1, -1, 1)).sum(
                2
            )

        # 4. prepare output
        avg_obs_embed = obs_embeds.mean(-2)
        extras = {}
        for aux_uuid in self.auxiliary_uuids:
            extras[aux_uuid] = {
                "beliefs": aux_to_fused_memory[aux_uuid],
                "obs_embeds": avg_obs_embed,
                "aux_model": self.aux_models[aux_uuid],
            }

        rel_sensor_out = observations[self.relationship_sensor_uuid]
        (
            cls_embedding,
            relationship_embeddings,
        ) = self.relationship_transformer.forward(
            e0=rel_sensor_out["e0"],
            e1=rel_sensor_out["e1"],
            i0=rel_sensor_out["i0"],
            i1=rel_sensor_out["i1"],
            relationship_type=rel_sensor_out["relationship_type"],
            memory=x,
        )

        actor_critic_output = ActorCriticOutput(
            distributions=self.actor(cls_embedding),
            values=self.critic(cls_embedding),
            extras=extras,
        )
        return actor_critic_output, memory
