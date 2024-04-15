import logging
from typing import List, Mapping, Optional

import tensorflow as tf
from seqio import (
    EncDecFeatureConverter,
    FeatureConverter,
    non_padding_position,
    utils,
)


def trim_and_pad_dataset(
    dataset: tf.data.Dataset,
    feature_lengths: Mapping[str, int],
    protected_features: List[str] = [],
) -> tf.data.Dataset:
    """Trim and pad first dimension of features to `feature_lengths`.

    Args:
    dataset: tf.data.Dataset, the dataset to trim/pad examples in.
    feature_lengths: map from feature key to final length. Other features will
        be returned unchanged.

    Returns:
    Trimmed/padded tf.data.Dataset.
    """

    def _trim_and_pad(k: str, t: tf.Tensor) -> tf.Tensor:
        """Trim/pad to the first axis of `t` to be of size `length`."""
        if k not in feature_lengths or k in protected_features:
            return t
        length_k = feature_lengths[k]
        t = t[:length_k]
        pad_amt = length_k - tf.shape(t)[0]
        padded_t = tf.pad(t, [(0, pad_amt)] + [(0, 0)] * (len(t.shape) - 1))
        padded_t.set_shape([length_k] + t.shape.as_list()[1:])
        return padded_t

    return dataset.map(
        lambda x: {k: _trim_and_pad(k, t) for k, t in x.items()},
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


def trim_and_pack_dataset(
    dataset: tf.data.Dataset,
    feature_lengths: Mapping[str, int],
    use_custom_ops: bool = False,
    protected_features: List[str] = [],
) -> tf.data.Dataset:
    """Creates a 'packed' version of a dataset on-the-fly.

    Modified from the tensor2tensor library.

    This is meant to replace the irritation of having to create a separate
    "packed" version of a dataset to train efficiently on TPU.

    Each example in the output dataset represents several examples in the
    input dataset.

    For each key in the input dataset that also exists in `feature_lengths`, two
    additional keys are created:
    <key>_segment_ids: an int32 tensor identifying the parts
        representing the original example.
    <key>_positions: an int32 tensor identifying the position within the
        original example.

    Features that are not in `feature_lengths` will be removed.

    Example:
    Two input examples get combined to form an output example.
    The input examples are:
    {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0], "idx": 0}
    {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1], "idx": 1}
    The output example is:
    {
                    "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
        "inputs_segment_ids": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
            "inputs_positions": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                    "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
        "targets_segment_ids": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
        "targets_positions": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
    }

    0 represents padding in both the inputs and the outputs.

    Sequences in the incoming examples are truncated to length in
    `feature_lengths`, and the sequences in the output examples all have this
    fixed (padded) length. Features not in `features_length` (i.e, "idx") are
    removed.

    Args:
    dataset: a tf.data.Dataset
    feature_lengths: map from feature key to final length. Other features will
        be discarded.
    use_custom_ops: a boolean - custom ops are faster but require a custom-built
        binary, which is not currently possible on cloud-tpu.

    Returns:
    a tf.data.Dataset
    """
    element_spec = dataset.element_spec
    # Make sure that the dataset contains all keys in `feature_lengths`.
    for k in feature_lengths:
        if k not in element_spec:
            raise ValueError(
                f"Feature '{k}' not found in dataset. Available keys are "
                f"{list(element_spec.keys())}"
            )
    if (
        not element_spec[k].shape.is_compatible_with(tf.TensorShape([None]))
        and not use_custom_ops
    ):
        raise ValueError(
            f"Features to be packed must be one-dimensional. '{k}' is not.' "
            "Consider setting use_custom_ops if you have higher-rank features."
        )

    # Warn if there are any additional keys that will be removed.
    additional_keys = set(element_spec) - set(feature_lengths)
    if additional_keys:
        logging.warning(
            "Features not in `features_length` will be removed during packing: %s",
            additional_keys,
        )

    ds = dataset.map(
        lambda x: {
            k: x[k] if k in protected_features else x[k][:l, ...]
            for k, l in feature_lengths.items()
        },
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Setting batch_size=length ensures that the concatenated sequences (if they
    # have length >=1) are sufficient to fill at least one packed example.
    batch_size = max(feature_lengths.values())
    padded_shapes = {k: [-1] for k in feature_lengths}
    for k in feature_lengths:
        padded_shapes[k].extend(dataset.element_spec[k].shape[1:])
    ds = ds.padded_batch(batch_size, padded_shapes=padded_shapes)

    # temporarily remove protected features from feature length
    protected_features_lengths = {
        k: feature_lengths[k] for k in protected_features
    }
    feature_lengths = {
        k: feature_lengths[k]
        for k in feature_lengths
        if k not in protected_features
    }

    if use_custom_ops:
        ds = utils._pack_with_custom_ops(ds, feature_lengths)
    else:
        ds = utils._pack_with_tf_ops(ds, feature_lengths)

    # Set the Tensor shapes correctly since they get lost in the process.
    def _set_shape(x):
        for k, v in x.items():
            new_shape = [feature_lengths[utils._strip_packed_feature_key(k)]]
            new_shape.extend(v.get_shape()[1:])
            v.set_shape(new_shape)
        return x

    ret = ds.map(_set_shape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    feature_lengths.update(protected_features_lengths)
    return ret


class MemoryQAFeatureConverter(EncDecFeatureConverter):
    TASK_FEATURES = {
        "memory": FeatureConverter.FeatureSpec(dtype=tf.float32),
        "inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
    }
    NON_PAD_OR_PACK_FEATURES = ["memory"]
    MODEL_FEATURES = {
        "encoder_input_memory": FeatureConverter.FeatureSpec(dtype=tf.float32),
        "encoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
    }
    PACKING_FEATURE_DTYPES = {
        "encoder_segment_ids": tf.int32,
        "decoder_segment_ids": tf.int32,
        "encoder_positions": tf.int32,
        "decoder_positions": tf.int32,
    }

    def _pack_or_pad(
        self, ds: tf.data.Dataset, packed_lengths: Mapping[str, int]
    ) -> tf.data.Dataset:
        """Trim/pad to packed_lengths and optionally pack the input dataset.
        The difference between this implementation and the one in the base class is that we protect
        protected arguments from being processed. This is helpful when using continuous features.
        """
        if self.pack:
            raise NotImplementedError(
                "Packing is not implemented for MemoryQAFeatureConverter"
            )
            ds = trim_and_pack_dataset(
                ds,
                packed_lengths,
                self._use_custom_packing_ops,
                self.NON_PAD_OR_PACK_FEATURES,
            )
        else:
            ds = trim_and_pad_dataset(
                ds, packed_lengths, self.NON_PAD_OR_PACK_FEATURES
            )
        return ds

    def _convert_features(
        self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
    ) -> tf.data.Dataset:
        """Convert the dataset to be fed to the encoder-decoder model.

        The conversion process involves two steps

        1. Each feature in the `task_feature_lengths` is trimmed/padded and
        optionally packed depending on the value of self.pack.
        2. "inputs" fields are mapped to the encoder input and "targets" are mapped
        to decoder input (after being shifted) and target.

        All the keys in the `task_feature_lengths` should be present in the input
        dataset, which may contain some extra features that are not in the
        `task_feature_lengths`. They will not be included in the output dataset.
        One common scenario is the "inputs_pretokenized" and "targets_pretokenized"
        fields.

        Args:
        ds: an input tf.data.Dataset to be converted.
        task_feature_lengths: a mapping from feature to its length.

        Returns:
        ds: the converted dataset.
        """

        def convert_example(
            features: Mapping[str, tf.Tensor]
        ) -> Mapping[str, tf.Tensor]:
            # targets_segment_id is present only for a packed dataset.
            decoder_input_tokens = utils.make_autoregressive_inputs(
                features["targets"],
                sequence_id=features.get("targets_segment_ids", None),
            )

            d = {
                "encoder_input_tokens": features["inputs"],
                "decoder_target_tokens": features["targets"],
                "decoder_input_tokens": decoder_input_tokens,
                # Loss is computed for all but the padding positions.
                "decoder_loss_weights": non_padding_position(
                    features["targets"]
                ),
            }

            if self.pack:
                d["encoder_segment_ids"] = features["inputs_segment_ids"]
                d["decoder_segment_ids"] = features["targets_segment_ids"]
                d["encoder_positions"] = features["inputs_positions"]
                d["decoder_positions"] = features["targets_positions"]

            d["encoder_input_memory"] = features["memory"]
            return d

        ds = self._pack_or_pad(ds, task_feature_lengths)
        return ds.map(
            convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    def get_model_feature_lengths(
        self, task_feature_lengths: Mapping[str, int]
    ) -> Mapping[str, int]:
        """Define the length relationship between input and output features."""
        model_feature_lengths = super().get_model_feature_lengths(
            task_feature_lengths
        )
        model_feature_lengths["encoder_input_memory"] = task_feature_lengths[
            "memory"
        ]
        return model_feature_lengths
