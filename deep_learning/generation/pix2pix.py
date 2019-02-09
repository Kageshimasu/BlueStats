import tensorflow as tf
from common.base import BaseModel
from util.layers import *

class Pix2Pix(BaseModel):

    def __init__(self, session, size=(256, 256), channel=3, function_generator=None, function_discriminator=None):
        super().__init__()
        self._inputs = tf.clip_by_value(tf.placeholder(tf.float32, (None, size[0], size[1], channel)), 1e-12, 1.0)
        self._teacher = tf.clip_by_value(tf.placeholder(tf.float32, (None, size[0], size[1], channel)), 1e-12, 1.0)
        self._is_training = tf.placeholder(tf.bool)
        self._session = session
        generator = self._predict if function_generator == None else function_generator
        discriminator = self._discriminator if function_discriminator == None else function_discriminator
        self._logits_generator = generator(self._inputs, self._is_training, channel=channel)
        self._logits_discriminator_fake = discriminator(self._logits_generator, self._is_training)
        self._logits_discriminator_real = discriminator(self._teacher, self._is_training, reuse=True)
        self._loss_generator, self._loss_discriminator = self._adversarial_loss(
            self._teacher,
            self._logits_generator,
            self._logits_discriminator_fake,
            self._logits_discriminator_real,
        )
        vars_t = tf.trainable_variables()
        vars_generator = [v for v in vars_t if 'g_' in v.name]
        vars_discriminator = [v for v in vars_t if 'd_' in v.name]
        self._optimaizer_generator = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self._loss_generator, var_list=vars_generator)
        self._optimaizer_discriminator = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self._loss_discriminator, var_list=vars_discriminator)
        self._accuracy_generator = accuracy(self._logits_generator, self._teacher)
        self._session.run(tf.global_variables_initializer())
        
    def train(self, inputs, teacher):
        self._session.run(
            self._optimaizer_generator,
            feed_dict={
                self._inputs: inputs,
                self._teacher: teacher,
                self._is_training: True
                }
            )
        self._session.run(
            self._optimaizer_discriminator,
            feed_dict={
                self._inputs: inputs,
                self._teacher: teacher,
                self._is_training: True
                }
            )
    
    def predict(self, inputs):
        return self._session.run(
            self._logits_generator,
            feed_dict={
                self._inputs: inputs,
                self._is_training: True
                }
            )
    
    def evaluate(self, inputs, teacher):
        acc_gen = self._session.run(
            self._accuracy_generator,
            feed_dict={
                self._inputs: inputs,
                self._teacher: teacher,
                self._is_training: True
                }
            )
        loss_gen = self._session.run(
            self._loss_generator,
            feed_dict={
                self._inputs: inputs,
                self._teacher: teacher,
                self._is_training: True
                }
            )
        loss_dis = self._session.run(
            self._loss_discriminator,
            feed_dict={
                self._inputs: inputs,
                self._teacher: teacher,
                self._is_training: True
                }
            )
        evaluated_list = {
            "acc_gen": acc_gen,
            "loss_gen" : loss_gen,
            "loss_dis": loss_dis,
            }
        return evaluated_list
    
    def save(self, file_path):
        saver = tf.train.Saver()
        saver.save(self._session, file_path)

    def load(self, file_path):
        saver = tf.train.Saver()
        saver.restore(self._session, file_path)
    
    def _adversarial_loss(self, teacher, logits_generator, logits_discriminator_fake, logits_discriminator_real):
        loss_discriminator = -tf.reduce_mean(
            # the discriminator tries to discriminate a real one as 1 from a fake one as 0
            tf.log(logits_discriminator_real + 1e-12) + tf.log(1 - logits_discriminator_fake + 1e-12)
            )
        loss_generator = -tf.reduce_mean(
            # the generator tries to make the discriminator distinguish a fake one as 1
            tf.log(logits_discriminator_fake + 1e-12)
            )
        reconstruction_error = tf.reduce_mean(
            tf.abs(teacher - logits_generator)
            )
        weight_l1 = tf.constant(100.0, dtype=tf.float32)
        loss_generator += reconstruction_error * weight_l1
        return loss_generator, loss_discriminator

    def _predict(self, inputs, is_training, channel=3, sampling_num=12):
        logits_generator = None
        logits_discriminator = None
        num_scope_name = 0
        with tf.variable_scope("generator") as scope:
            filters = 64
            downsampled = []
            for i in range(1, sampling_num + 1):
                if i % 3 == 0 and i != 0:
                    inputs = max_pool(inputs)
                    filters *= 2
                    continue
                inputs = conv(inputs, filters, is_training=is_training, scope_name="g_" + str(num_scope_name))
                num_scope_name += 1
                if i % 2 == 0 and i != 0:
                    downsampled.append(inputs) # append it to concatenate
            upsampled = inputs
            for i in range(1, sampling_num + 1):
                if i % 3 == 0 and i != 0:
                    filters = int(filters // 2)
                    upsampled = tf.concat([deconv(upsampled, filters, scope_name="g_" + str(num_scope_name)), downsampled.pop()], axis=3)
                    num_scope_name += 1
                    continue
                upsampled = conv(upsampled, filters, is_training=is_training, scope_name="g_" + str(num_scope_name))
                num_scope_name += 1
            segmap = upsampled
            for i in range(1, 3):
                segmap = conv(segmap, filters, is_training=is_training, scope_name="g_" + str(num_scope_name))
                num_scope_name += 1
            return conv(segmap, channel, kernel_size=1, activation=tf.nn.tanh, scope_name="g_output")

    def _discriminator(self, inputs, is_training, reuse=False):
        logits_discriminator = None
        num_scope_name = 0
        with tf.variable_scope("discriminator") as scope:
            filters = 64
            if reuse:
                scope.reuse_variables()
            for i in range(1, 3):
                inputs = conv(inputs, filters * i, is_training=is_training, scope_name="d_" + str(num_scope_name))
                num_scope_name += 1
                inputs = max_pool(inputs)
            inputs = global_average_pooling(inputs, filters * 4, scope_name="d_" + str(num_scope_name))
            logits_discriminator = dense(inputs, 1, activation=tf.nn.sigmoid, scope_name="d_output")
        return logits_discriminator