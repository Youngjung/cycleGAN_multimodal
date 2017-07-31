# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import time
import datetime
import numpy as np
import tensorflow as tf

import configuration
import show_and_tell_model
from inference_utils import vocabulary, caption_generator

import pdb

FLAGS = tf.app.flags.FLAGS

class BehaviorGenerator(object):
	def __init__(self, config, vocab):
		self.config = config
		self.vocab = vocab

	def build(self):
		config = self.config
		vocab = self.vocab

		# Build the model (teacher-forcing mode).
		model_teacher = show_and_tell_model.ShowAndTellModel(
				config, mode="train", train_inception=FLAGS.train_inception)
		model_teacher.build()

		# Build the model (free-running mode).
		model_free = show_and_tell_model.ShowAndTellModel(
				config, mode="free", train_inception=FLAGS.train_inception,
				vocab=vocab, reuse=True )
		model_free.build([model_teacher.images,model_teacher.input_seqs,model_teacher.target_seqs,model_teacher.input_mask])

		# Build the model for validation with variable sharing
		model_valid = show_and_tell_model.ShowAndTellModel(
				config, mode="inference", reuse=True )
		model_valid.build()
		self.sampler = caption_generator.CaptionGenerator( model_valid, vocab )

		# get teacher behavior
		teacher_outputs, [teacher_state_c,teacher_state_h] = model_teacher.behavior
		teacher_state_c = tf.expand_dims( teacher_state_c, axis=1 )
		teacher_state_h = tf.expand_dims( teacher_state_h, axis=1 )
		teacher_behavior =tf.concat([teacher_outputs,teacher_state_c,teacher_state_h],axis=1)

		# get free behavior
		free_outputs, [free_state_c,free_state_h] = model_free.behavior
		free_state_c = tf.expand_dims( free_state_c, axis=1 )
		free_state_h = tf.expand_dims( free_state_h, axis=1 )
		free_behavior = tf.concat( [free_outputs,free_state_c,free_state_h], axis=1 )
		
		# summary
		summary = {}
		summary['NLL_loss'] = tf.summary.scalar('NLL_loss', model_teacher.total_loss)

		# set outputs
		self.teacher_behavior = teacher_behavior
		self.free_behavior = free_behavior
		self.input_mask = model_teacher.input_mask # mask is not an output but required in im2txt discriminator
		self.input_image = model_teacher.images # image is not an output but required in text2image discriminator

		# NLL loss
		self.loss = model_teacher.total_loss

		self.summary = summary
		self.free_sentence = model_free.free_sentence
		self.teacher_sentence = model_teacher.input_seqs

		# misc
		self.inception_variables = model_teacher.inception_variables
		self.global_step = model_teacher.global_step
		self.init_fn = model_teacher.init_fn

	def generate_text(self, sess, image_valid):
		return self.sampler.beam_search( sess, image_valid )
