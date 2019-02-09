import tensorflow as tf

def conv(inputs, filters, kernel_size=(3, 3), activation=tf.nn.relu, is_training=False):
  conved = tf.layers.conv2d(
    inputs=inputs,
    filters=filters,
    kernel_size=kernel_size,
    padding="same",
    activation=activation
  )
  return batch_normalization(conved, is_training)

def deconv(inputs, filters, kernel_size=(2, 2), strides=(2, 2), activation=tf.nn.relu, is_training=False):
  return tf.layers.conv2d_transpose(
    inputs=inputs,
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding="same",
    activation=activation
  )

def max_pool(inputs, size=(2, 2), strides=2):
  return tf.layers.max_pooling2d(
    inputs=inputs,
    pool_size=size,
    strides=strides
  )

def batch_normalization(inputs, is_training):
  return tf.layers.batch_normalization(
    inputs=inputs,
    axis=-1,
    momentum=9e-1,
    epsilon=1e-3,
    center=True,
    scale=True,
    training=is_training
  )

def cross_entropy(teacher, logits):
  return -tf.reduce_mean(teacher * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))

def pixel_wise_softmax(logits):
  exponential_map = tf.exp(logits - tf.reduce_max(logits, axis=3, keepdims=True))
  return exponential_map / tf.reduce_sum(exponential_map, axis=3, keepdims=True)