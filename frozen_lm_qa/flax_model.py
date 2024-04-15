import copy
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from transformers import FlaxT5PreTrainedModel, T5Config
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxSeq2SeqLMOutput,
)
from transformers.models.t5.modeling_flax_t5 import (
    FlaxT5EncoderModule,
    FlaxT5ForConditionalGeneration,
    FlaxT5Stack,
)

from frozen_lm_qa.conditional_generation_flax_utils import (
    ConditionalFlaxGenerationMixin,
)
from frozen_lm_qa.network import PromptGeneration


class T5ConfigWithMemory(T5Config):
    def __init__(self, *args, **kwargs):
        self.inp_features = kwargs.pop("inp_features", None)
        self.n_prompt_tokens = kwargs.pop("n_prompt_tokens", None)
        super().__init__(*args, **kwargs)


class FlaxT5StackWithMemory(FlaxT5Stack):
    prompt_generator: PromptGeneration = None

    def __call__(
        self,
        memory_input,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        init_cache: bool = False,
    ):
        hidden_states = self.embed_tokens(input_ids)
        # Edit begin: added memory_input
        hidden_states = jnp.concatenate(
            [
                self.prompt_generator(
                    memory_input, deterministic=deterministic
                ),
                hidden_states,
            ],
            axis=-2,
        )
        # Edit end
        hidden_states = self.dropout(
            hidden_states, deterministic=deterministic
        )

        outputs = self.block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
        )

        hidden_states = outputs[0]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(
            hidden_states, deterministic=deterministic
        )

        # Add last layer
        all_hidden_states = None

        if output_hidden_states:
            all_hidden_states = outputs.hidden_states
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            if output_hidden_states:
                return (
                    hidden_states,
                    all_hidden_states,
                ) + outputs[2:]
            return (hidden_states,) + outputs[1:]

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class FlaxT5ForMemoryQuestionAnsweringModule(nn.Module):
    config: T5ConfigWithMemory
    dtype: jnp.dtype = jnp.float32

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder

    def setup(self):
        self.model_dim = self.config.d_model

        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(
                self.config.initializer_factor
            ),
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        # Edit begin: added prompt_generator
        self.prompt_generator = PromptGeneration(
            inp_features=self.config.inp_features,
            features=self.config.d_model,
            n_prompt_tokens=self.config.n_prompt_tokens,
            dtype=self.dtype,
            name="prompt_generator",
        )
        self.encoder = FlaxT5StackWithMemory(
            encoder_config,
            self.shared,
            dtype=self.dtype,
            prompt_generator=self.prompt_generator,
        )
        # Edit end

        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.config.num_decoder_layers
        self.decoder = FlaxT5Stack(
            decoder_config, self.shared, dtype=self.dtype
        )

        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_factor
            ),
            dtype=self.dtype,
        )

    def __call__(
        self,
        # Edit begin: added memory_input
        memory_input,
        # Edit end
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        deterministic: bool = True,
    ):
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        # Encode
        encoder_outputs = self.encoder(
            # Edit begin: added memory_input
            memory_input=memory_input,
            # Edit end
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        if self.config.tie_word_embeddings:
            shared_embedding = self.shared.variables["params"]["embedding"]
            lm_logits = self.lm_head.apply(
                {"params": {"kernel": shared_embedding.T}}, sequence_output
            )
        else:
            lm_logits = self.lm_head(sequence_output)

        if not return_dict:
            return (lm_logits,) + decoder_outputs[1:] + encoder_outputs

        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class FlaxT5ForMemoryQuestionAnsweringPreTrainedModel(FlaxT5PreTrainedModel):
    def init_weights(
        self,
        rng: jax.random.PRNGKey,
        input_shape: Tuple,
        params: FrozenDict = None,
    ) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")

        # Edit begin: added memory_input
        memory_input = jnp.zeros(
            (input_shape[0], self.config.inp_features), dtype="f4"
        )

        attention_mask = jnp.ones_like(
            jnp.zeros(
                (input_shape[0], self.config.n_prompt_tokens + input_shape[1]),
                dtype="i4",
            )
        )
        args = [memory_input, input_ids, attention_mask]
        # Edit end
        if self.module_class not in [FlaxT5EncoderModule]:
            decoder_input_ids = jnp.ones_like(input_ids)
            decoder_attention_mask = jnp.ones_like(input_ids)
            args.extend([decoder_input_ids, decoder_attention_mask])

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs,
            *args,
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
        self,
        # Edit begin: added memory_input
        memory_input: jnp.ndarray,
        # Edit end
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_input_ids: jnp.ndarray = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        if decoder_input_ids is None:
            raise ValueError(
                "Make sure to provide both `input_ids` and `decoder_input_ids`. `decoder_input_ids` is not passed"
                " here."
            )

        # prepare encoder inputs
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        attention_mask = jnp.concatenate(
            [
                jnp.ones((memory_input.shape[0], self.config.n_prompt_tokens)),
                attention_mask,
            ],
            axis=1,
        )

        # prepare decoder inputs
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            memory_input=jnp.array(memory_input, dtype=self.dtype),
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(
                decoder_attention_mask, dtype="i4"
            ),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
        )


class FlaxT5ForMemoryQuestionAnswering(
    ConditionalFlaxGenerationMixin,
    FlaxT5ForMemoryQuestionAnsweringPreTrainedModel,
    FlaxT5ForConditionalGeneration,
):
    module_class = FlaxT5ForMemoryQuestionAnsweringModule
