"""Generation utils with additional vector inputs.
This is a modified version of transformers/generation_flax_utils.py
"""
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from transformers import FlaxLogitsProcessorList
from transformers.generation_flax_utils import FlaxGenerationMixin


class ConditionalFlaxGenerationMixin(FlaxGenerationMixin):
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in
    [`FlaxPreTrainedModel`].

    Inherited methods:
        1. _run_loop_in_debug
        2. _expand_to_num_beams
        3. _adapt_logits_for_beam_search
        4. _get_logits_warper
        5. _get_logits_processor
        6. _greey_search
        7. _sample
        8. _greedy_search

    The class exposes [`~generation_flax_utils.FlaxGenerationMixin.generate`], which can be used for:
            - *greedy decoding* by calling [`~generation_flax_utils.FlaxGenerationMixin._greedy_search`] if
              `num_beams=1` and `do_sample=False`.
            - *multinomial sampling* by calling [`~generation_flax_utils.FlaxGenerationMixin._sample`] if `num_beams=1`
              and `do_sample=True`.
            - *beam-search decoding* by calling [`~generation_utils.FlaxGenerationMixin._beam_search`] if `num_beams>1`
              and `do_sample=False`.
    """

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, memory_input, input_ids, params, model_kwargs
    ):
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not (
                argument.startswith("decoder_")
                or argument.startswith("cross_attn")
            )
        }
        model_kwargs["encoder_outputs"] = self.encode(
            memory_input,
            input_ids,
            params=params,
            return_dict=True,
            **encoder_kwargs,
        )
        return model_kwargs

    def generate(
        self,
        memory_input: jnp.ndarray,
        input_ids: jnp.ndarray,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        do_sample: Optional[bool] = None,
        prng_key: Optional[jnp.ndarray] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        num_beams: Optional[int] = None,
        no_repeat_ngram_size: Optional[int] = None,
        min_length: Optional[int] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head. The method supports the following
        generation methods for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

            - *greedy decoding* by calling [`~generation_flax_utils.FlaxGenerationMixin._greedy_search`] if
              `num_beams=1` and `do_sample=False`.
            - *multinomial sampling* by calling [`~generation_flax_utils.FlaxGenerationMixin._sample`] if `num_beams=1`
              and `do_sample=True`.
            - *beam-search decoding* by calling [`~generation_utils.FlaxGenerationMixin._beam_search`] if `num_beams>1`
              and `do_sample=False`.

        <Tip warning={true}>

        Apart from `inputs`, all the arguments below will default to the value of the attribute of the same name as
        defined in the model's config (`config.json`) which in turn defaults to the
        [`~modeling_utils.PretrainedConfig`] of the model.

        </Tip>

        Most of these parameters are explained in more detail in [this blog
        post](https://huggingface.co/blog/how-to-generate).

        Parameters:

            input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            temperature (`float`, *optional*, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (`int`, *optional*, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher
                are kept for generation.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            decoder_start_token_id (`int`, *optional*):
                If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            trace (`bool`, *optional*, defaults to `True`):
                Whether to trace generation. Setting `trace=False` should only be used for debugging and will lead to a
                considerably slower runtime.
            params (`Dict[str, jnp.ndarray]`, *optional*):
                Optionally the model parameters can be passed. Can be useful for parallelized generation.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If the model
                is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs
                should be prefixed with *decoder_*. Also accepts `encoder_outputs` to skip encoder part.

        Return:
            [`~utils.ModelOutput`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        >>> model = FlaxAutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> input_context = "The dog"
        >>> # encode input context
        >>> input_ids = tokenizer(input_context, return_tensors="np").input_ids
        >>> # generate candidates using sampling
        >>> outputs = model.generate(input_ids=input_ids, max_length=20, top_k=30, do_sample=True)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ```"""
        # set init values
        max_length = (
            max_length if max_length is not None else self.config.max_length
        )
        min_length = (
            min_length if min_length is not None else self.config.min_length
        )
        bos_token_id = (
            bos_token_id
            if bos_token_id is not None
            else self.config.bos_token_id
        )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.config.eos_token_id
        )
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.decoder_start_token_id
        )
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        if decoder_start_token_id is None and self.config.is_encoder_decoder:
            raise ValueError(
                "`decoder_start_token_id` has to be defined for encoder-decoder generation."
            )
        if min_length is not None and min_length > max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({min_length}) is larger than the maximum "
                f"length ({max_length})"
            )

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            if model_kwargs.get("encoder_outputs") is None:
                model_kwargs = (
                    self._prepare_encoder_decoder_kwargs_for_generation(
                        memory_input, input_ids, params, model_kwargs
                    )
                )
            # prepare decoder_input_ids for generation
            input_ids = (
                jnp.ones((input_ids.shape[0], 1), dtype="i4")
                * decoder_start_token_id
            )

        do_sample = (
            do_sample if do_sample is not None else self.config.do_sample
        )
        num_beams = (
            num_beams if num_beams is not None else self.config.num_beams
        )

        if not do_sample and num_beams == 1:
            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size,
                min_length,
                max_length,
                eos_token_id,
                forced_bos_token_id,
                forced_eos_token_id,
            )
            return self._greedy_search(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif do_sample and num_beams == 1:
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature
            )
            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size,
                min_length,
                max_length,
                eos_token_id,
                forced_bos_token_id,
                forced_eos_token_id,
            )
            return self._sample(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                prng_key,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif not do_sample and num_beams > 1:
            # broadcast input_ids & encoder_outputs
            input_ids = self._expand_to_num_beams(
                input_ids, num_beams=num_beams
            )

            if "encoder_outputs" in model_kwargs:
                model_kwargs["encoder_outputs"][
                    "last_hidden_state"
                ] = self._expand_to_num_beams(
                    model_kwargs["encoder_outputs"]["last_hidden_state"],
                    num_beams=num_beams,
                )

            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = self._expand_to_num_beams(
                    model_kwargs["attention_mask"], num_beams=num_beams
                )

            logits_processor = self._get_logits_processor(
                no_repeat_ngram_size,
                min_length,
                max_length,
                eos_token_id,
                forced_bos_token_id,
                forced_eos_token_id,
            )

            return self._beam_search(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        else:
            raise NotImplementedError(
                "`Beam sampling is currently not implemented."
            )
