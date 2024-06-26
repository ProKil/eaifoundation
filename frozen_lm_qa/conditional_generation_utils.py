"""Generation utils with additional vector inputs.
This is a modified version of transformers/generation_flax_utils.py
"""
import inspect
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import torch
from transformers import (
    BeamSearchScorer,
    ConstrainedBeamSearchScorer,
    Constraint,
    DisjunctiveConstraint,
    LogitsProcessorList,
    PhrasalConstraint,
    StoppingCriteriaList,
)
from transformers.generation_utils import (
    BeamSampleOutput,
    BeamSearchOutput,
    GenerationMixin,
    GreedySearchOutput,
    SampleOutput,
)
from transformers.utils import ModelOutput, logging

logger = logging.get_logger(__name__)


class ConditionalGenerationMixin(GenerationMixin):
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

    def _prepare_attention_mask_for_generation(
        self,
        memory_input,
        inputs: torch.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[int],
    ) -> torch.LongTensor:
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [
            torch.int,
            torch.long,
        ]
        is_pad_token_in_inputs = (pad_token_id is not None) and (
            pad_token_id in inputs
        )
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        # Check if input is input_ids and padded -> only then is attention_mask defined
        if (
            is_input_ids
            and is_pad_token_in_inputs
            and is_pad_token_not_equal_to_eos_token_id
        ):
            return torch.cat(
                [
                    torch.ones(
                        (inputs.shape[0], self.config.n_prompt_tokens)
                    ).long(),
                    inputs.ne(pad_token_id).long(),
                ],
                dim=1,
            )
        else:
            return torch.cat(
                [
                    torch.ones(
                        (inputs.shape[0], self.config.n_prompt_tokens)
                    ).long(),
                    torch.ones(
                        inputs.shape[:2],
                        dtype=torch.long,
                        device=inputs.device,
                    ),
                ],
                dim=1,
            )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        memory_input: torch.Tensor,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        encoder_kwargs["memory_input"] = memory_input

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = (
            model_input_name
            if model_input_name is not None
            else self.main_input_name
        )
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(
            **encoder_kwargs
        )

        return model_kwargs

    @torch.no_grad()
    def generate(
        self,
        memory_input: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        force_words_ids: Optional[
            Union[Iterable[int], Iterable[Iterable[int]]]
        ] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        logits_processor: Optional[
            LogitsProcessorList
        ] = LogitsProcessorList(),
        renormalize_logits: Optional[bool] = None,
        stopping_criteria: Optional[
            StoppingCriteriaList
        ] = StoppingCriteriaList(),
        constraints: Optional[List[Constraint]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        exponential_decay_length_penalty: Optional[
            Tuple[Union[int, float]]
        ] = None,
        **model_kwargs,
    ) -> Union[
        GreedySearchOutput,
        SampleOutput,
        BeamSearchOutput,
        BeamSampleOutput,
        torch.LongTensor,
    ]:
        # 1. Set generation parameters if not already defined
        bos_token_id = (
            bos_token_id
            if bos_token_id is not None
            else self.config.bos_token_id
        )
        num_beams = (
            num_beams if num_beams is not None else self.config.num_beams
        )
        length_penalty = (
            length_penalty
            if length_penalty is not None
            else self.config.length_penalty
        )
        early_stopping = (
            early_stopping
            if early_stopping is not None
            else self.config.early_stopping
        )
        num_beam_groups = (
            num_beam_groups
            if num_beam_groups is not None
            else self.config.num_beam_groups
        )
        do_sample = (
            do_sample if do_sample is not None else self.config.do_sample
        )
        num_return_sequences = (
            num_return_sequences
            if num_return_sequences is not None
            else self.config.num_return_sequences
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

        if eos_token_id is None and hasattr(self.config, "decoder"):
            eos_token_id = self.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            logger.warning(
                f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation."
            )
            pad_token_id = eos_token_id

        output_scores = (
            output_scores
            if output_scores is not None
            else self.config.output_scores
        )
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
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        (
            inputs_tensor,
            model_input_name,
            model_kwargs,
        ) = self._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        accepts_attention_mask = "attention_mask" in set(
            inspect.signature(self.forward).parameters.keys()
        )
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if (
            model_kwargs.get("attention_mask", None) is None
            and requires_attention_mask
            and accepts_attention_mask
        ):
            model_kwargs[
                "attention_mask"
            ] = self._prepare_attention_mask_for_generation(
                memory_input, inputs_tensor, pad_token_id, eos_token_id
            )

        if (
            self.config.is_encoder_decoder
            and "encoder_outputs" not in model_kwargs
        ):
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                memory_input, inputs_tensor, model_kwargs, model_input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
                device=inputs_tensor.device,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        input_ids_seq_length = input_ids.shape[-1]

        # 5. Prepare `max_length` depending on other stopping criteria
        # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids_seq_length
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set "
                f"but they serve the same purpose. `max_length` {max_length} "
                f"will take priority over `max_new_tokens` {max_new_tokens}.",
                UserWarning,
            )
        # default to config if still None
        max_length = (
            max_length if max_length is not None else self.config.max_length
        )
        min_length = (
            min_length if min_length is not None else self.config.min_length
        )

        if min_length is not None and min_length > max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({min_length}) is larger than the maximum "
                f"length ({max_length})"
            )
        if input_ids_seq_length >= max_length:
            input_ids_string = (
                "decoder_input_ids"
                if self.config.is_encoder_decoder
                else "input_ids"
            )
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but ``max_length`` is set to"
                f" {max_length}. This can lead to unexpected behavior. You should consider increasing"
                " ``config.max_length`` or ``max_length``."
            )

        # 6. determine generation mode
        is_constraint_gen_mode = (
            constraints is not None or force_words_ids is not None
        )
        is_greedy_gen_mode = (
            (num_beams == 1)
            and (num_beam_groups == 1)
            and do_sample is False
            and not is_constraint_gen_mode
        )
        is_sample_gen_mode = (
            (num_beams == 1)
            and (num_beam_groups == 1)
            and do_sample is True
            and not is_constraint_gen_mode
        )
        is_beam_gen_mode = (
            (num_beams > 1)
            and (num_beam_groups == 1)
            and do_sample is False
            and not is_constraint_gen_mode
        )
        is_beam_sample_gen_mode = (
            (num_beams > 1)
            and (num_beam_groups == 1)
            and do_sample is True
            and not is_constraint_gen_mode
        )
        is_group_beam_gen_mode = (
            (num_beams > 1)
            and (num_beam_groups > 1)
            and not is_constraint_gen_mode
        )

        if num_beam_groups > num_beams:
            raise ValueError(
                "`num_beam_groups` has to be smaller or equal to `num_beams`"
            )
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            logits_processor=logits_processor,
            renormalize_logits=renormalize_logits,
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length,
            max_time=max_time,
            stopping_criteria=stopping_criteria,
        )

        # 9. go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # 10. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                temperature=temperature,
                num_beams=num_beams,
                renormalize_logits=renormalize_logits,
            )

            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`."
                )

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now."
                )

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 12. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                temperature=temperature,
                num_beams=num_beams,
                renormalize_logits=renormalize_logits,
            )

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now."
                )
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * num_return_sequences,
                num_beams=num_beams,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`."
                )

            if num_beams % num_beam_groups != 0:
                raise ValueError(
                    "`num_beams` should be divisible by `num_beam_groups` for group beam search."
                )

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now."
                )

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                max_length=stopping_criteria.max_length,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 12. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`."
                )

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now."
                )

            if num_beams <= 1:
                raise ValueError(
                    "`num_beams` needs to be greater than 1 for constrained genertation."
                )

            if do_sample:
                raise ValueError(
                    "`do_sample` needs to be false for constrained generation."
                )

            if num_beam_groups is not None and num_beam_groups > 1:
                raise ValueError(
                    "`num_beam_groups` not supported yet for constrained generation."
                )

            final_constraints = []
            if constraints is not None:
                final_constraints = constraints

            if force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {force_words_ids}."
                    )

                if (
                    not isinstance(force_words_ids, list)
                    or len(force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in force_words_ids:
                    if isinstance(word_ids[0], list):
                        if (
                            not isinstance(word_ids, list)
                            or len(word_ids) == 0
                        ):
                            typeerror()
                        if any(
                            not isinstance(token_ids, list)
                            for token_ids in word_ids
                        ):
                            typeerror()
                        if any(
                            any(
                                (not isinstance(token_id, int) or token_id < 0)
                                for token_id in token_ids
                            )
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if (
                            not isinstance(word_ids, list)
                            or len(word_ids) == 0
                        ):
                            typeerror()
                        if any(
                            (not isinstance(token_id, int) or token_id < 0)
                            for token_id in word_ids
                        ):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 10. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=num_beams,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 12. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )
