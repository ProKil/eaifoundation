"""Model for frozen language model and question answering
Code are mainly adapted from t5x/models.py
"""

import functools
from typing import Any, Mapping, MutableMapping, Optional, Tuple, Union

import flax
import gin
import jax
import jax.numpy as jnp
from chex import PyTreeDef
from flax.core import scope as flax_scope

from t5x import decoding
from t5x.models import Array, EncoderDecoderModel


class MemoryQAModel(EncoderDecoderModel):
    def _compute_logits(
        self,
        params: PyTreeDef,
        batch: Mapping[str, jnp.ndarray],
        dropout_rng: Optional[jax.random.KeyArray] = None,
        mutable: flax_scope.CollectionFilter = False,
        other_variables: Optional[PyTreeDef] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
        """Computes logits via a forward pass of `self.module_cls`."""
        # Dropout is provided only for the training mode.
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else None
        if other_variables is None:
            other_variables = {}
        return self.module.apply(
            {"params": params, **other_variables},
            batch["encoder_input_memory"],
            batch["encoder_input_tokens"],
            batch["decoder_input_tokens"],
            batch["decoder_target_tokens"],
            encoder_segment_ids=batch.get("encoder_segment_ids", None),
            decoder_segment_ids=batch.get("decoder_segment_ids", None),
            encoder_positions=batch.get("encoder_positions", None),
            decoder_positions=batch.get("decoder_positions", None),
            decode=False,
            enable_dropout=rngs is not None,
            rngs=rngs,
            mutable=mutable,
        )

    def _compute_logits_from_slice(
        self,
        flat_ids: jnp.ndarray,
        flat_cache: Mapping[str, jnp.ndarray],
        params: PyTreeDef,
        encoded_inputs: jnp.ndarray,
        raw_inputs: jnp.ndarray,
        raw_input_memory: jnp.ndarray,
        max_decode_length: int,
    ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Token slice to logits from decoder model."""
        # flat_ids: [batch * beam, seq_len=1]
        # cache is expanded inside beam_search to become flat_cache
        # flat_cache: [batch * beam, num_heads, depth_per_head, max_decode_len]
        # flat_logits: [batch * beam, seq_len=1, vocab]
        flat_logits, new_vars = self.module.apply(
            {"params": params, "cache": flat_cache},
            encoded_inputs,
            raw_input_memory,
            raw_inputs,  # only needed for encoder padding mask
            flat_ids,
            flat_ids,
            enable_dropout=False,
            decode=True,
            max_decode_length=max_decode_length,
            mutable=["cache"],
            method=self.module.decode,
        )
        # Remove sequence length dimension since it's always 1 during decoding.
        flat_logits = jnp.squeeze(flat_logits, axis=1)
        new_flat_cache = new_vars["cache"]
        return flat_logits, new_flat_cache

    def predict_batch_with_aux(
        self,
        params: PyTreeDef,
        batch: Mapping[str, jnp.ndarray],
        rng: Optional[jax.random.KeyArray] = None,
        decoder_params: Optional[MutableMapping[str, Any]] = None,
        return_all_decodes: bool = False,
        num_decodes: int = 1,
        prompt_with_targets: bool = False,
    ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        inputs = batch["encoder_input_tokens"]
        input_memory = batch["encoder_input_memory"]
        # [batch, target_len]
        target_shape = batch["decoder_input_tokens"].shape
        target_type = batch["decoder_input_tokens"].dtype
        _, variables_with_cache = self.module.apply(
            {"params": params},
            jnp.ones(input_memory.shape, dtype=input_memory.dtype),
            jnp.ones(inputs.shape, inputs.dtype),
            jnp.ones(target_shape, target_type),
            jnp.ones(target_shape, target_type),
            decode=True,
            enable_dropout=False,
            mutable=["cache"],
        )

        cache = variables_with_cache["cache"]
        encoded_inputs = decoding.flat_batch_beam_expand(
            self.module.apply(
                {"params": params},
                input_memory,
                inputs,
                enable_dropout=False,
                method=self.module.encode,
            ),
            num_decodes,
        )

        raw_inputs = decoding.flat_batch_beam_expand(inputs, num_decodes)
        raw_input_memory = decoding.flat_batch_beam_expand(
            input_memory, num_decodes
        )

        tokens_ids_to_logits = functools.partial(
            self._compute_logits_from_slice,
            params=params,
            encoded_inputs=encoded_inputs,
            raw_inputs=raw_inputs,
            raw_input_memory=raw_input_memory,
            max_decode_length=target_shape[1],
        )

        if decoder_params is None:
            decoder_params = {}
        if rng is not None:
            if decoder_params.get("decode_rng") is not None:
                raise ValueError(
                    f"Got RNG both from the `rng` argument ({rng}) and "
                    f"`decoder_params['decode_rng']` ({decoder_params['decode_rng']}). "
                    "Please specify one or the other."
                )
            decoder_params["decode_rng"] = rng

        if prompt_with_targets:
            decoder_prompt_inputs = batch["decoder_input_tokens"]
            decoder_prompt_inputs = decoder_prompt_inputs * (
                decoder_prompt_inputs != self.output_vocabulary.eos_id
            )
        else:
            decoder_prompt_inputs = jnp.zeros_like(
                batch["decoder_input_tokens"]
            )

        scanned = (
            hasattr(self.module, "scan_layers") and self.module.scan_layers
        )

        if "eos_id" not in decoder_params:
            decoder_params["eos_id"] = self.output_vocabulary.eos_id
        decodes, scores = self._decode_fn(
            inputs=decoder_prompt_inputs,
            cache=cache,
            tokens_to_logits=tokens_ids_to_logits,
            num_decodes=num_decodes,
            cache_offset=1 if scanned else 0,
            **decoder_params,
        )

        # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
        # in increasing order of log-probability.
        # Return the highest scoring beam sequence.
        if return_all_decodes:
            return decodes, {"scores": scores}
        else:
            return decodes[:, -1, :], {"scores": scores[:, -1]}

    def get_initial_variables(
        self,
        rng: jax.random.KeyArray,
        input_shapes: Mapping[str, Array],
        input_types: Optional[Mapping[str, jnp.dtype]] = None,
    ) -> flax_scope.FrozenVariableDict:
        """Get the initial variables for an encoder-decoder model."""
        input_types = {} if input_types is None else input_types
        memory_shape = input_shapes["encoder_input_memory"]
        memory_type = input_types.get("encoder_input_memory", jnp.float32)
        encoder_shape = input_shapes["encoder_input_tokens"]
        encoder_type = input_types.get("encoder_input_tokens", jnp.float32)
        decoder_shape = input_shapes["decoder_input_tokens"]
        decoder_type = input_types.get("decoder_input_tokens", jnp.float32)
        if "encoder_positions" in input_shapes:
            encoder_positions = jnp.ones(
                input_shapes["encoder_positions"],
                input_types.get("encoder_positions", jnp.int32),
            )
        else:
            encoder_positions = None
        if "decoder_positions" in input_shapes:
            decoder_positions = jnp.ones(
                input_shapes["decoder_positions"],
                input_types.get("decoder_positions", jnp.int32),
            )
        else:
            decoder_positions = None
        if "encoder_segment_ids" in input_shapes:
            encoder_segment_ids = jnp.ones(
                input_shapes["encoder_segment_ids"],
                input_types.get("encoder_segment_ids", jnp.int32),
            )
        else:
            encoder_segment_ids = None
        if "decoder_segment_ids" in input_shapes:
            decoder_segment_ids = jnp.ones(
                input_shapes["decoder_segment_ids"],
                input_types.get("decoder_segment_ids", jnp.int32),
            )
        else:
            decoder_segment_ids = None
        initial_variables = self.module.init(
            rng,
            jnp.ones(memory_shape, memory_type),
            jnp.ones(encoder_shape, encoder_type),
            jnp.ones(decoder_shape, decoder_type),
            jnp.ones(decoder_shape, decoder_type),
            encoder_positions=encoder_positions,
            decoder_positions=decoder_positions,
            encoder_segment_ids=encoder_segment_ids,
            decoder_segment_ids=decoder_segment_ids,
            decode=False,
            enable_dropout=False,
        )
        return initial_variables
