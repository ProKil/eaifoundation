import tensorflow as tf


# Read the data back out.
def decode_fn(record_bytes):
    return tf.train.Example.FromString(record_bytes.numpy())


for batch in tf.data.TFRecordDataset(["probing_qa.tfrecord"]):
    print(decode_fn(batch))
