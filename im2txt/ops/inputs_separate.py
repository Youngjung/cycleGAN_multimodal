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

"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from ops import inputs as input_ops
from ops import image_processing


def parse_sequence_example(serialized, image_feature, caption_feature):
	"""Parses a tensorflow.SequenceExample into an image and caption.

	Args:
		serialized: A scalar string Tensor; a single serialized SequenceExample.
		image_feature: Name of SequenceExample context feature containing image
			data.
		caption_feature: Name of SequenceExample feature list containing integer
			captions.

	Returns:
		encoded_image: A scalar string Tensor containing a JPEG encoded image.
		caption: A 1-D uint64 Tensor with dynamically specified length.
	"""
	context, sequence = tf.parse_single_sequence_example(
			serialized,
			context_features={
					image_feature: tf.FixedLenFeature([], dtype=tf.string)
			},
			sequence_features={
					caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
			})

	encoded_image = context[image_feature]
	caption = sequence[caption_feature]
	return encoded_image, caption

class MSCOCORunner(object):
	"""
		This class manages the the background threads needed to fill
				a queue full of data.
		"""
	def __init__(self,
					config,
					is_training,
					shard_queue_name="filename_queue",
					value_queue_name="input_queue"):
		"""Prefetches string values from disk into an input queue.
	
		In training the capacity of the queue is important because a larger queue
		means better mixing of training examples between shards. The minimum number of
		values kept in the queue is values_per_shard * input_queue_capacity_factor,
		where input_queue_memory factor should be chosen to trade-off better mixing
		with memory usage.
	
		Args:
			reader: Instance of tf.ReaderBase.
			file_pattern: Comma-separated list of file patterns (e.g.
					/tmp/train_data-?????-of-00100).
			is_training: Boolean; whether prefetching for training or eval.
			batch_size: Model batch size used to determine queue capacity.
			values_per_shard: Approximate number of values per shard.
			input_queue_capacity_factor: Minimum number of values to keep in the queue
				in multiples of values_per_shard. See comments above.
			num_reader_threads: Number of reader threads to fill the queue.
			shard_queue_name: Name for the shards filename queue.
			value_queue_name: Name for the values input queue.
	
		Returns:
			A Queue containing prefetched string values.
		"""
		self.config = config
		self.is_training = is_training
		file_pattern = config.input_file_pattern
		batch_size = config.batch_size
		values_per_shard = config.values_per_input_shard
		input_queue_capacity_factor = config.input_queue_capacity_factor or 16
		num_reader_threads = config.num_input_reader_threads or 1
		self.reader = tf.TFRecordReader()

		data_files = []
		for pattern in file_pattern.split(","):
			data_files.extend(tf.gfile.Glob(pattern))
		if not data_files:
			tf.logging.fatal("Found no input files matching %s", file_pattern)
		else:
			tf.logging.info("Prefetching values from %d files matching %s",
											len(data_files), file_pattern)
	
		if is_training:
			filename_queue = tf.train.string_input_producer(
					data_files, shuffle=True, capacity=16, name=shard_queue_name)
			min_queue_examples = values_per_shard * input_queue_capacity_factor
			capacity = min_queue_examples + 100 * batch_size
			values_queue = tf.RandomShuffleQueue(
					capacity=capacity,
					min_after_dequeue=min_queue_examples,
					dtypes=[tf.string],
					name="random_" + value_queue_name)
		else:
			filename_queue = tf.train.string_input_producer(
					data_files, shuffle=False, capacity=1, name=shard_queue_name)
			capacity = values_per_shard + 3 * batch_size
			values_queue = tf.FIFOQueue(
					capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)
	
		enqueue_ops = []
		for _ in range(num_reader_threads):
			_, value = self.reader.read(filename_queue)
			enqueue_ops.append(values_queue.enqueue([value]))
		tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
				values_queue, enqueue_ops))
		tf.summary.scalar(
				"queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
				tf.cast(values_queue.size(), tf.float32) * (1. / capacity))
	
		self.values_queue = values_queue

	# this is from show_and_tell_model.py
	def get_inputs(self):
		images_and_captions = []
		for thread_id in range(self.config.num_preprocess_threads):
			serialized_sequence_example = self.values_queue.dequeue()
			encoded_image, caption = input_ops.parse_sequence_example(
					serialized_sequence_example,
					image_feature=self.config.image_feature_name,
					caption_feature=self.config.caption_feature_name)
			image = self.process_image(encoded_image, thread_id=thread_id)
			images_and_captions.append([image, caption])

		# Batch inputs.
		queue_capacity = (2 * self.config.num_preprocess_threads *
											self.config.batch_size)
		images, input_seqs, target_seqs, input_mask = (
				input_ops.batch_with_dynamic_pad(images_and_captions,
														 batch_size=self.config.batch_size,
														 queue_capacity=queue_capacity))
		return images, input_seqs, target_seqs, input_mask

	def process_image(self, encoded_image, thread_id=0):
		"""Decodes and processes an image string.
	
		Args:
			encoded_image: A scalar string Tensor; the encoded image.
			thread_id: Preprocessing thread id used to select the ordering of color
				distortions.
	
		Returns:
			A float32 Tensor of shape [height, width, 3]; the processed image.
		"""
		return image_processing.process_image(encoded_image,
											is_training=self.is_training,
											height=self.config.image_height,
											width=self.config.image_width,
											thread_id=thread_id,
											image_format=self.config.image_format)

