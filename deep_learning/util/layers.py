import tensorflow as tf

def dense(inputs, units, activation=tf.nn.leaky_relu, is_training=False, scope_name=None):
  return tf.layers.dense(
    inputs,
    units,
    activation=activation,
    name=scope_name
  )

def conv(inputs, filters, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, bn=False, is_training=False, scope_name=None):
  conved = tf.layers.conv2d(
    inputs=inputs,
    filters=filters,
    strides=strides,
    kernel_size=kernel_size,
    padding="same",
    activation=activation,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4),
    name=scope_name
  )
  return batch_normalization(conved, is_training) if bn == True else conved

def deconv(inputs, filters, kernel_size=2, strides=2, activation=tf.nn.leaky_relu, is_training=False, scope_name=None):
  return tf.layers.conv2d_transpose(
    inputs=inputs,
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding="same",
    activation=activation,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4),
    name=scope_name
  )

def global_average_pooling(inputs, num_class, activation=tf.nn.relu, scope_name=""):
  inputs = conv(inputs, num_class, strides=2, activation=activation, scope_name=scope_name)
  for _ in range(2):
      inputs = tf.reduce_mean(inputs, axis=1)
  return inputs

def max_pool(inputs, size=(2, 2), strides=2):
  return tf.layers.max_pooling2d(
    inputs=inputs,
    pool_size=size,
    strides=strides
  )

def average_pool(inputs, size=(2, 2), strides=2):
  return tf.layers.average_pooling2d(
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

def accuracy(inputs, teacher):
  correct_prediction = tf.equal(tf.argmax(inputs, 1), tf.argmax(teacher, 1))
  return tf.reduce_mean(tf.cast(correct_prediction, "float32"))