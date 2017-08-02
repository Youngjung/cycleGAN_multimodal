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
from behavior_generator import BehaviorGenerator
from behavior_discriminator import BehaviorDiscriminator
#from text2image_generator import Text2ImageGenerator
#from text2image_discriminator import Text2ImageDiscriminator
from inference_utils import vocabulary, caption_generator

from ops.inputs_separate import MSCOCORunner

import pdb

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
						"File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("inception_checkpoint_file", "",
						"Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("train_dir", "",
						"Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_inception", False,
						"Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
						"Frequency at which loss and global step are logged.")
tf.flags.DEFINE_string("vocab_file", "", "")

tf.logging.set_verbosity(tf.logging.INFO)



def main(unused_argv):
	assert FLAGS.input_file_pattern, "--input_file_pattern is required"
	assert FLAGS.train_dir, "--train_dir is required"

	# Create training directory.
	train_dir = FLAGS.train_dir
	filename_saved_model = os.path.join(FLAGS.train_dir,'im2txt')
	if not tf.gfile.IsDirectory(train_dir):
		tf.logging.info("Creating training directory: %s", train_dir)
		tf.gfile.MakeDirs(train_dir)

	save_flags( os.path.join(FLAGS.train_dir,'flags.txt') )

	model_config = configuration.ModelConfig()
	model_config.input_file_pattern = FLAGS.input_file_pattern
	model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file
	training_config = configuration.TrainingConfig()

	vocab = vocabulary.Vocabulary( FLAGS.vocab_file )

	# Build the TensorFlow graph.
	g = tf.Graph()
	with g.as_default():
		summary = {}

		# im2txt generator part
		im2txt_generator = BehaviorGenerator( model_config, vocab )
		im2txt_generator.build()
		NLL_loss = im2txt_generator.loss
		global_step = im2txt_generator.global_step
		free_sentence = im2txt_generator.free_sentence
		teacher_sentence = im2txt_generator.teacher_sentence

		# prepare behavior to be LSTM's input
		teacher_behavior = im2txt_generator.teacher_behavior
		free_behavior = im2txt_generator.free_behavior
		summary['NLL_loss'] = tf.summary.scalar('NLL_loss', im2txt_generator.loss)

		# temporal debugging variables
		free_lstm_outputs = im2txt_generator.free_lstm_outputs
		free_lstm_final_state = im2txt_generator.free_lstm_final_state
		teacher_lstm_outputs = im2txt_generator.teacher_lstm_outputs
		teacher_lstm_final_state = im2txt_generator.teacher_lstm_final_state

		# collect LSTM feature from generator
#		generated_text_feature = free_behavior[:-3]

		# im2txt discriminator part
		im2txt_discriminator = BehaviorDiscriminator( model_config )
		im2txt_discriminator.build( teacher_behavior, free_behavior, im2txt_generator.input_mask )
		d_loss = im2txt_discriminator.d_loss
		d_loss_teacher = im2txt_discriminator.d_loss_teacher
		d_loss_free = im2txt_discriminator.d_loss_free
		g_loss = im2txt_discriminator.g_loss
		d_accuracy = im2txt_discriminator.accuracy
		summary['d_merged'] = im2txt_discriminator.summary_merged

#		# text2image generator part
#		t2i_generator = Text2ImageGenerator()
#		t2i_generator.build( generated_text_feature )
#
#		# collect fake/real images
#		fake_image = t2i_generator.fake_image
#		real_image = im2txt_generator.input_image
#
#		# text2image discriminator part
#		t2i_discriminator = Text2ImageDiscriminator()
#		t2i_discriminator.build( real_image, fake_image )
		
		g_and_NLL_loss = g_loss + NLL_loss
		summary['g_and_NLL_loss'] = tf.summary.scalar('g_and_NLL_loss',g_and_NLL_loss)
		summary['all'] = tf.summary.merge_all()

		# Set up the learning rate for training ops
		learning_rate_decay_fn = None
		if FLAGS.train_inception:
			learning_rate = tf.constant(training_config.train_inception_learning_rate)
		else:
			learning_rate = tf.constant(training_config.initial_learning_rate)
			if training_config.learning_rate_decay_factor > 0:
				num_batches_per_epoch = (training_config.num_examples_per_epoch //
																 model_config.batch_size)
				decay_steps = int(num_batches_per_epoch *
													training_config.num_epochs_per_decay)

				def _learning_rate_decay_fn(_learning_rate, _global_step):
					return tf.train.exponential_decay(
							_learning_rate,
							_global_step,
							decay_steps=decay_steps,
							decay_rate=training_config.learning_rate_decay_factor,
							staircase=True)

				learning_rate_decay_fn = _learning_rate_decay_fn

		# Collect trainable variables
		vars_all = [ v for v in tf.trainable_variables() \
								 if v not in im2txt_generator.inception_variables ]
		d_vars = [ v for v in vars_all if 'discr' in v.name ]
		g_vars = [ v for v in vars_all if 'discr' not in v.name ]

		# Set up the training ops.
		train_op_NLL = tf.contrib.layers.optimize_loss(
											loss = NLL_loss,
											global_step = global_step,
											learning_rate = learning_rate,
											optimizer = training_config.optimizer,
											clip_gradients = training_config.clip_gradients,
											learning_rate_decay_fn = learning_rate_decay_fn,
											variables = g_vars,
											name='optimize_NLL_loss' )

		train_op_disc_teacher = tf.contrib.layers.optimize_loss(
											loss = d_loss_teacher,
											global_step = global_step,
											learning_rate = learning_rate,
											optimizer = training_config.optimizer,
											clip_gradients = training_config.clip_gradients,
											learning_rate_decay_fn = learning_rate_decay_fn,
											variables = d_vars,
											name='optimize_disc_loss' )

		train_op_disc_free = tf.contrib.layers.optimize_loss(
											loss = d_loss_free,
											global_step = global_step,
											learning_rate = learning_rate,
											optimizer = training_config.optimizer,
											clip_gradients = training_config.clip_gradients,
											learning_rate_decay_fn = learning_rate_decay_fn,
											variables = d_vars,
											name='optimize_disc_loss' )

		train_op_gen = tf.contrib.layers.optimize_loss(
											loss = g_and_NLL_loss,
											global_step=global_step,
											learning_rate=learning_rate,
											optimizer='Adam',
											clip_gradients=training_config.clip_gradients,
											learning_rate_decay_fn=learning_rate_decay_fn,
											variables = g_vars,
											name='optimize_gen_loss' )



		# Set up the Saver for saving and restoring model checkpoints.
		saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

		with tf.Session() as sess:
			nBatches = num_batches_per_epoch
			summaryWriter = tf.summary.FileWriter(train_dir, sess.graph)

			# initialize all variables
			tf.global_variables_initializer().run()

			# load inception variables
			im2txt_generator.init_fn( sess )
			
			# start input enqueue threads
			with tf.device("/cpu:0"):
				coco_runner = MSCOCORunner( model_config, is_training=True )
				t_images, t_input_seqs, t_target_seqs, t_input_mask = coco_runner.get_inputs()
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)
						
			counter = 0
			start_time = time.time()
			could_load, checkpoint_counter = load( sess, saver, train_dir )
			if could_load:
				counter = checkpoint_counter

			try:
				# for validation
				f_valid_text = open(os.path.join(train_dir,'valid.txt'),'a')
				filenames = os.listdir('testimgs')
				filenames.sort()
				valid_images = []
				print( 'validation image filenames' )
				for filename in filenames:
					with tf.gfile.GFile(os.path.join('testimgs', filename),'r') as f:
						valid_images.append( f.read() )
					print( filename )

			
				# run inference for not-trained model
#				f_valid_text.write( 'initial caption (beam) {}\n'.format( str(datetime.datetime.now().time())[:-7] ) )
#				for i, valid_image in enumerate(valid_images):
#					captions = im2txt_generator.sampler.beam_search( sess, valid_image )
#					for j, caption in enumerate(captions):
#						sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
#						sentence = " ".join(sentence)
#						sentence = "  {}-{}) {} (p={:.8f})".format(i+1,j+1, sentence, math.exp(caption.logprob))
#						print( sentence )
#						f_valid_text.write( sentence +'\n' )
#				f_valid_text.flush()

				# run training loop
				lossnames_to_print = ['NLL_loss','g_loss', 'd_loss', 'd_acc', 'g_acc']
				val_NLL_loss = float('Inf')
				val_g_loss = float('Inf')
				val_d_loss = float('Inf')
				val_d_acc = 0
				val_g_acc = 0
				b_G_pretrain_done = False
				for epoch in range(FLAGS.number_of_steps):
					for batch_idx in range(nBatches):
						counter += 1
						images, input_seqs, target_seqs, input_mask = sess.run( 
							[t_images, t_input_seqs, t_target_seqs, t_input_mask] )
						generator_feed_dict = { im2txt_generator.images:images,
												im2txt_generator.input_seqs:input_seqs,
												im2txt_generator.target_seqs:target_seqs,
												im2txt_generator.input_mask:input_mask } 


#						if True:
						if not b_G_pretrain_done and val_NLL_loss> 3:
							_, val_NLL_loss, summary_str, val_free_sentence, val_teacher_sentence = sess.run(
									[train_op_NLL, NLL_loss, summary['NLL_loss'], free_sentence, teacher_sentence],
									feed_dict=generator_feed_dict )
							summaryWriter.add_summary(summary_str, counter)
						else:
							b_G_pretrain_done = True

							# train discriminator
							_, val_d_loss, val_d_acc, summary_str, val_teacher_behavior, val_free_behavior, \
							val_free_lstm_outputs, val_free_lstm_final_state, val_teacher_lstm_outputs, val_teacher_lstm_final_state \
							= sess.run([train_op_disc_teacher, d_loss, d_accuracy, summary['d_merged'], teacher_behavior, free_behavior,
							free_lstm_outputs, free_lstm_final_state, teacher_lstm_outputs, teacher_lstm_final_state],
									feed_dict=generator_feed_dict )
							pdb.set_trace()
							summaryWriter.add_summary(summary_str, counter)
							_, val_d_loss, val_d_acc, summary_str \
							= sess.run([train_op_disc_free, d_loss, d_accuracy, summary['d_merged']],
									feed_dict=generator_feed_dict )
							summaryWriter.add_summary(summary_str, counter)

							# train generator
							# val_g_acc is temporarily named variable instead of val_d_acc
							_, val_g_loss, val_NLL_loss, val_g_acc, summary_str = sess.run( 
								[train_op_gen,g_loss,NLL_loss, d_accuracy, summary['g_and_NLL_loss']], #summary['all']],
									feed_dict=generator_feed_dict )
							summaryWriter.add_summary(summary_str, counter)
							_, val_g_loss, val_NLL_loss, val_g_acc, summary_str = sess.run( 
								[train_op_gen,g_loss,NLL_loss, d_accuracy, summary['g_and_NLL_loss']], #summary['all']],
									feed_dict=generator_feed_dict )
							summaryWriter.add_summary(summary_str, counter)
			
						if counter % FLAGS.log_every_n_steps==0:
							elapsed = time.time() - start_time
							log( epoch, batch_idx, nBatches, lossnames_to_print,
								 [val_NLL_loss,val_g_loss,val_d_loss,val_d_acc,val_g_acc], elapsed, counter )
			
						if counter % 500 == 1 or \
							(epoch==FLAGS.number_of_steps-1 and batch_idx==nBatches-1) :
							saver.save( sess, filename_saved_model, global_step=counter)
			
						if (batch_idx+1) % (nBatches//50) == 0  or batch_idx == nBatches-1:
							# run test after every epoch
							f_valid_text.write( 'count {} epoch {} batch {}/{} ({})\n'.format( \
									counter, epoch, batch_idx, nBatches, str(datetime.datetime.now().time())[:-7] ) )
							for i, valid_image in enumerate(valid_images):
								captions = im2txt_generator.sampler.beam_search( sess, valid_image )
								for j, caption in enumerate(captions):
									sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
									sentence = " ".join(sentence)
									sentence = "  {}-{}) {} (p={:.8f})".format(i+1,j+1, sentence, math.exp(caption.logprob))
									print( sentence )
									f_valid_text.write( sentence +'\n' )
							# free sentence check
							for i, caption in enumerate(val_free_sentence):
								if i>9: break
								sentence = [vocab.id_to_word(w) for w in caption[1:-1]]
								sentence = " ".join(sentence)
								sentence = "  free %d) %s" % (i+1, sentence)
								print( sentence )
								f_valid_text.write( sentence +'\n' )
								sentence = [vocab.id_to_word(w) for w in val_teacher_sentence[i,1:]]
								sentence = " ".join(sentence)
								sentence = "  teacher %d) %s" % (i+1, sentence)
								print( sentence )
								f_valid_text.write( sentence +'\n' )
							f_valid_text.flush()
			
			except tf.errors.OutOfRangeError:
				print('Finished training: epoch limit reached')
			finally:
				coord.request_stop()
			coord.join(threads)

def load(sess, saver, checkpoint_dir):
	import re
	print(" [*] Reading checkpoints...")

	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
		counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
		print(" [*] Success to read {}".format(ckpt_name))
		return True, counter
	else:
		print(" [*] Failed to find a checkpoint")
		return False, 0

def log( epoch, batch, nBatches, lossnames, losses, elapsed, counter=None, filelogger=None ):
	nDigits = len(str(nBatches))
	str_lossnames = ""
	str_losses = ""
	assert( len(lossnames) == len(losses) )
	isFirst = True
	for lossname, loss in zip(lossnames,losses):
		if not isFirst:
			str_lossnames += ','
			str_losses += ', '
		str_lossnames += lossname
		if type(loss) == str:
			str_losses += loss
		else:
			str_losses += '{:.4f}'.format(loss)
		isFirst = False

	m,s = divmod( elapsed, 60 )
	h,m = divmod( m,60 )
	timestamp = "{:2}:{:02}:{:02}".format( int(h),int(m),int(s) )
	log = "{} e{} b {:>{}}/{} ({})=({})".format( timestamp, epoch, batch, nDigits, nBatches, str_lossnames, str_losses )
	if counter is not None:
		log = "{:>5}_".format(counter) + log
	print( log )
	if filelogger:
		filelogger.write( log )
	return log

def save_flags( path ):
    flags_dict = tf.flags.FLAGS.__flags
    with open(path, 'w') as f:
        for key,val in flags_dict.iteritems():
            f.write( '{} = {}\n'.format(key,val) )

if __name__ == "__main__":
	tf.app.run()


