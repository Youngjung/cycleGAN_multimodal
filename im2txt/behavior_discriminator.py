import math
import os
import time
import datetime
import numpy as np
import tensorflow as tf

import configuration
import behavior_discriminator

import pdb

FLAGS = tf.app.flags.FLAGS

class BehaviorDiscriminator(object):
	def __init__(self, config):
		self.config = config
		self.initializer = tf.random_uniform_initializer(
       							 minval=-self.config.initializer_scale,
       							 maxval=self.config.initializer_scale)

	# arguments = inputs to the network
	def build(self, behavior_teacher, behavior_free, input_mask):
		config = self.config

		lstm_cell = tf.contrib.rnn.BasicLSTMCell(config.num_lstm_units)
		lstm_cell = tf.contrib.rnn.DropoutWrapper(
							lstm_cell,
							input_keep_prob=config.lstm_dropout_keep_prob,
							output_keep_prob=config.lstm_dropout_keep_prob)

		with tf.variable_scope("discriminator") as scope_disc:
			teacher_lengths = tf.reduce_sum( input_mask, 1 )+2
			free_lengths = tf.ones_like(teacher_lengths)*(30+2)

			# run lstm
			outputs_teacher, _ = tf.nn.dynamic_rnn( cell=lstm_cell,
												inputs = behavior_teacher,
												sequence_length = teacher_lengths,
												dtype = tf.float32,
												scope = scope_disc )

			# gather last outputs (deals with variable length of captions)
			teacher_lengths = tf.expand_dims( teacher_lengths, 1 )
			batch_range = tf.expand_dims(tf.constant( np.array(range(config.batch_size)),dtype=tf.int32 ),1)
			gather_idx = tf.concat( [batch_range,teacher_lengths-1], axis=1 )
			last_output_teacher = tf.gather_nd( outputs_teacher, gather_idx )

			# FC to get T/F logits
			logits_teacher = tf.contrib.layers.fully_connected( inputs = last_output_teacher,
															num_outputs = 2,
															activation_fn = None,
															weights_initializer = self.initializer,
															scope = scope_disc )

			scope_disc.reuse_variables()
			outputs_free, _ = tf.nn.dynamic_rnn( cell=lstm_cell,
												inputs = behavior_free,
												sequence_length = free_lengths,
												dtype = tf.float32,
												scope = scope_disc )
			last_output_free = outputs_free[:,-1,:]
			logits_free = tf.contrib.layers.fully_connected( inputs = last_output_free,
															num_outputs = 2,
															activation_fn = None,
															weights_initializer = self.initializer,
															scope = scope_disc )

		# loss
		d_loss_teacher = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(name='loss_teacher',
									 logits=logits_teacher, labels=tf.ones_like(logits_teacher) ) )
		d_loss_free = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(name='loss_free',
									 logits=logits_free, labels=tf.zeros_like(logits_free) ) )
		d_loss = d_loss_teacher + d_loss_free

		g_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(name='g_loss',
									 logits=logits_free, labels=tf.ones_like(logits_free) ) )

		# accuracy
		accuracy_teacher = tf.reduce_mean( tf.cast( tf.argmax( logits_teacher, axis=1 ), tf.float32 ) )
		accuracy_free = tf.reduce_mean( tf.cast( 1-tf.argmax( logits_free, axis=1 ), tf.float32 ) )
		accuracy = ( accuracy_teacher + accuracy_free ) /2

		# summary
		summary = {}
		summary['d_loss'] = tf.summary.scalar('d_loss', d_loss)
		summary['d_loss_teacher'] = tf.summary.scalar('d_loss_teacher', d_loss_teacher)
		summary['d_loss_free'] = tf.summary.scalar('d_loss_free', d_loss_free)
		summary['g_loss'] = tf.summary.scalar('g_loss', g_loss)
		summary['d_logits_free'] = tf.summary.histogram('d_logits_free', logits_free)
		summary['d_accuracy'] = tf.summary.histogram('d_accuracy', accuracy)
		summary['d_accuracy_teacher'] = tf.summary.histogram('d_accuracy_teacher', accuracy_teacher)
		summary['d_accuracy_free'] = tf.summary.histogram('d_accuracy_free', accuracy_free)

		self.d_loss = d_loss
		self.d_loss_teacher = d_loss_teacher
		self.d_loss_free = d_loss_free
		self.g_loss = g_loss
		self.accuracy = accuracy
		self.accuracy_teacher = accuracy_teacher
		self.accuracy_free = accuracy_free

		self.summary = summary
