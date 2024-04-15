import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import flax.linen as nn
from flax import struct
from jax import numpy as jnp

from t5x.examples.t5 import layers
from t5x.examples.t5.layers import (
    DenseGeneral,
    MlpBlock,
    _convert_to_activation_function,
    default_embed_init,
    with_sharding_constraint,
)
from t5x.examples.t5.network import (
    Decoder,
    Encoder,
    EncoderLayer,
    T5Config,
    Transformer,
)

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]


class MlpBlock2d(nn.Module):
    """The same as MlpBlock, but the input is 2d (batch, hidden_state)"""

    intermediate_dim: int = 2048
    activations: Sequence[Union[str, Callable]] = ("relu",)
    kernel_init: Initializer = nn.initializers.variance_scaling(
        1.0, "fan_in", "truncated_normal"
    )
    intermediate_dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(  # type: ignore
        self, inputs, decode: bool = False, deterministic: bool = False
    ):
        """Applies Transformer MlpBlock module."""
        # Iterate over specified MLP input activation functions.
        # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
        activations = []
        for idx, act_fn in enumerate(self.activations):
            dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
            x = DenseGeneral(
                self.intermediate_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,  # type: ignore
                kernel_axes=("embed", "mlp"),
                name=dense_name,
            )(inputs)
            x = _convert_to_activation_function(act_fn)(x)
            activations.append(x)

        # Take elementwise product of above intermediate activations.
        x = functools.reduce(operator.mul, activations)
        # Apply dropout and final dense output projection.
        x = nn.Dropout(
            rate=self.intermediate_dropout_rate, broadcast_dims=(-2,)
        )(
            x, deterministic=deterministic
        )  # Broadcast along length.
        x = with_sharding_constraint(x, ("batch", "mlp"))
        output = DenseGeneral(
            inputs.shape[-1],
            dtype=self.dtype,
            kernel_init=self.kernel_init,  # type: ignore
            kernel_axes=("mlp", "embed"),
            name="wo",
        )(x)
        return output


class PromptGeneration(nn.Module):
    inp_features: int
    features: int
    n_prompt_tokens: int
    dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, input: Array, deterministic=False):  # type: ignore
        x = DenseGeneral(
            self.features * self.n_prompt_tokens,
            dtype=self.dtype,
            kernel_axes=("embed", "mlp"),
            name="wi",
        )(input)
        x = MlpBlock2d(self.features * self.n_prompt_tokens, dtype=self.dtype)(
            x, deterministic=deterministic
        )
        return x.reshape(input.shape[0], self.n_prompt_tokens, self.features)


class PromptTextEncoder(Encoder):
    """T5 Encoder with generated prompts."""

    config: T5Config
    shared_embedding: nn.Module
    prompt_generator: PromptGeneration

    @nn.compact
    def __call__(
        self,
        memory_input: Array,
        encoder_input_tokens,
        encoder_mask=None,
        deterministic=False,
    ):
        cfg = self.config
        assert memory_input.ndim == 2  # (batch_size, inp_features)
        rel_emb = layers.RelativePositionBiases(
            num_buckets=32,
            max_distance=128,
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            embedding_init=nn.initializers.variance_scaling(
                1.0, "fan_avg", "uniform"
            ),
            name="relpos_bias",
        )

        # [batch, length] -> [batch, length, emb_dim]
        x: Array = jnp.concatenate(
            [
                self.prompt_generator(
                    memory_input, deterministic=deterministic
                ),
                self.shared_embedding(encoder_input_tokens.astype("int32")),
            ],
            axis=-2,
        )
        x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic
        )
        x = x.astype(cfg.dtype)

        for lyr in range(cfg.num_encoder_layers):
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            x = EncoderLayer(
                config=cfg, relative_embedding=rel_emb, name=f"layers_{lyr}"
            )(x, encoder_mask, deterministic)

        x = layers.LayerNorm(dtype=cfg.dtype, name="encoder_norm")(x)
        return nn.Dropout(rate=cfg.dropout_rate)(
            x, deterministic=deterministic
        )


@struct.dataclass
class FrozenMemoryConfig:
    inp_features: int
    n_prompt_tokens: int


class FrozenTransformer(Transformer):
    """T5 Transformer with generated prompts."""

    config: T5Config
    frozen_memory_config: FrozenMemoryConfig

    def setup(self):
        cfg = self.config
        self.shared_embedding = layers.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.emb_dim,
            dtype=cfg.dtype,
            attend_dtype=jnp.float32,  # for logit training stability
            embedding_init=nn.initializers.normal(stddev=1.0),
            one_hot=True,
            name="token_embedder",
        )
        self.prompt_generator = PromptGeneration(
            inp_features=self.frozen_memory_config.inp_features,
            features=cfg.emb_dim,
            n_prompt_tokens=self.frozen_memory_config.n_prompt_tokens,
            dtype=cfg.dtype,
            name="prompt_generator",
        )
        self.encoder = PromptTextEncoder(
            config=cfg,
            shared_embedding=self.shared_embedding,
            prompt_generator=self.prompt_generator,
        )
        self.decoder = Decoder(
            config=cfg, shared_embedding=self.shared_embedding
        )

    def encode(
        self,
        encoder_input_memory: Array,
        encoder_input_tokens: Array,
        encoder_segment_ids=None,
        enable_dropout=True,
    ):
        """Applies Transformer encoder-branch on the inputs."""
        cfg = self.config
        fm_cfg = self.frozen_memory_config
        assert encoder_input_tokens.ndim == 2  # (batch, len)

        # Make padding attention mask.
        encoder_mask = layers.make_attention_mask(
            jnp.concatenate(
                [
                    jnp.ones(
                        (encoder_input_memory.shape[0], fm_cfg.n_prompt_tokens)
                    ),
                    encoder_input_tokens > 0,
                ],
                axis=1,
            ),
            jnp.concatenate(
                [
                    jnp.ones(
                        (encoder_input_memory.shape[0], fm_cfg.n_prompt_tokens)
                    ),
                    encoder_input_tokens > 0,
                ],
                axis=1,
            ),
            dtype=cfg.dtype,
        )
        # Add segmentation block-diagonal attention mask if using segmented data.
        if encoder_segment_ids is not None:
            encoder_mask = layers.combine_masks(
                encoder_mask,
                layers.make_attention_mask(
                    encoder_segment_ids,
                    encoder_segment_ids,
                    jnp.equal,
                    dtype=cfg.dtype,
                ),
            )

        return self.encoder(
            encoder_input_memory,
            encoder_input_tokens,
            encoder_mask,
            deterministic=not enable_dropout,
        )

    def decode(
        self,
        encoded,
        encoder_input_memory,
        encoder_input_tokens,  # only needed for masks
        decoder_input_tokens,
        decoder_target_tokens,
        encoder_segment_ids=None,
        decoder_segment_ids=None,
        decoder_positions=None,
        enable_dropout=True,
        decode=False,
        max_decode_length=None,
    ):
        """Applies Transformer decoder-branch on encoded-input and target."""
        cfg = self.config
        fm_cfg = self.frozen_memory_config

        # Make padding attention masks.
        if decode:
            # Do not mask decoder attention based on targets padding at
            # decoding/inference time.
            decoder_mask = None
            encoder_decoder_mask = layers.make_attention_mask(
                jnp.ones_like(decoder_target_tokens),
                jnp.concatenate(
                    [
                        jnp.ones(
                            (
                                encoder_input_memory.shape[0],
                                fm_cfg.n_prompt_tokens,
                            )
                        ),
                        encoder_input_tokens > 0,
                    ],
                    axis=1,
                ),
                dtype=cfg.dtype,
            )
        else:
            decoder_mask = layers.make_decoder_mask(
                decoder_target_tokens=decoder_target_tokens,
                dtype=cfg.dtype,
                decoder_segment_ids=decoder_segment_ids,
            )
            encoder_decoder_mask = layers.make_attention_mask(
                decoder_target_tokens > 0,
                jnp.concatenate(
                    [
                        jnp.ones(
                            (
                                encoder_input_memory.shape[0],
                                fm_cfg.n_prompt_tokens,
                            )
                        ),
                        encoder_input_tokens > 0,
                    ],
                    axis=1,
                ),
                dtype=cfg.dtype,
            )

        # Add segmentation block-diagonal attention masks if using segmented data.
        if encoder_segment_ids is not None:
            if decode:
                raise ValueError(
                    "During decoding, packing should not be used but "
                    "`encoder_segment_ids` was passed to `Transformer.decode`."
                )

        if decoder_segment_ids is not None and encoder_segment_ids is not None:
            encoder_decoder_mask = layers.combine_masks(
                encoder_decoder_mask,
                layers.make_attention_mask(
                    decoder_segment_ids,
                    encoder_segment_ids,
                    jnp.equal,
                    dtype=cfg.dtype,
                ),
            )

        logits = self.decoder(
            encoded,
            decoder_input_tokens=decoder_input_tokens,
            decoder_positions=decoder_positions,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
            deterministic=not enable_dropout,
            decode=decode,
            max_decode_length=max_decode_length,
        )
        return logits

    def __call__(
        self,
        encoder_input_memory,
        encoder_input_tokens,
        decoder_input_tokens,
        decoder_target_tokens,
        encoder_segment_ids=None,
        decoder_segment_ids=None,
        encoder_positions=None,
        decoder_positions=None,
        *,
        enable_dropout: bool = True,
        decode: bool = False,
    ):
        encoded = self.encode(
            encoder_input_memory,
            encoder_input_tokens,
            encoder_segment_ids=encoder_segment_ids,
            enable_dropout=enable_dropout,
        )

        return self.decode(
            encoded,
            encoder_input_memory,
            encoder_input_tokens,  # only used for masks
            decoder_input_tokens,
            decoder_target_tokens,
            encoder_segment_ids=encoder_segment_ids,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            enable_dropout=enable_dropout,
            decode=decode,
        )
