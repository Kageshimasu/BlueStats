import tensorflow as tf 
from common.base import BaseModel
from util.layers import *

class UNet(BaseModel):

  def __init__(self, session, class_num, size=(256, 256)):
    super().__init__()
    self._inputs = tf.placeholder(tf.float32, (None, size[0], size[1], 3))
    self._teacher = tf.placeholder(tf.float32, (None, size[0], size[1], 3))
    self._is_training = tf.placeholder(tf.bool)
    self._session = session
    self._logits = self._predict(self._inputs, class_num, self._is_training)
    self._cross_entropy = cross_entropy(
      tf.reshape(self._teacher, [-1, class_num]),
      tf.reshape(pixel_wise_softmax(self._logits), [-1, class_num])
    )
    self.optimaizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cross_entropy)
    self._session.run(tf.global_variables_initializer())

  def train(self, inputs, teacher):
    self._session.run(
      self.optimaizer,
      feed_dict={
        self._inputs: inputs,
        self._teacher: teacher,
        self.is_training: True
        }
      )
  
  def predict(self, inputs):
    return self._session.run(
      self._logits,
      feed_dict={
        self._inputs: inputs,
        self._is_training: True
      }
    )
  
  def evaluate(self, inputs, teacher):
    loss = self._session.run(
      self._cross_entropy,
      feed_dict={
        self._inputs: inputs,
        self._teacher: teacher,
        self._is_training: True
      }
    )
    iou = self._session.run(
      self._cross_entropy, # TODO: evaluate iou
      feed_dict={
        self._inputs: inputs,
        self._teacher: teacher,
        self._is_training: True
      }
    )
    evaluated_list = {
      "loss": loss,
      "iou" : iou,
    }
    return evaluated_list
  
  def save(self, file_path):
    saver = tf.train.Saver()
    saver.save(self._session, file_path)

  def load(self, file_path):
    saver = tf.train.Saver()
    saver.restore(self._session, file_path)

  def _predict(self, inputs, class_num, is_training, sampling_num=12):
    filters = 64
    downsampled = []
    # downsample
    for i in range(1, sampling_num + 1):
      if i % 3 == 0 and i != 0:
        inputs = max_pool(inputs)
        filters *= 2
        continue
      inputs = conv(inputs, filters, is_training=is_training)
      if i % 2 == 0 and i != 0:
        downsampled.append(inputs) # append it to concatenate
    # upsample
    upsampled = inputs
    for i in range(1, sampling_num + 1):
      if i % 3 == 0 and i != 0:
        filters = int(filters // 2)
        upsampled = tf.concat([deconv(upsampled, filters), downsampled.pop()], axis=3)
        continue
      upsampled = conv(upsampled, filters, is_training=is_training)
    # output segmentation map
    segmap = upsampled
    for i in range(1, 4):
      if i % 3 == 0 and i != 0:
        return conv(segmap, class_num)
      segmap = conv(segmap, filters, is_training=is_training)