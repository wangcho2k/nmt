# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions specifically for NMT."""
from __future__ import print_function

#import codecs
import time

import tensorflow as tf

from ..utils import evaluation_utils
from ..utils import misc_utils as utils

__all__ = ["decode_and_evaluate", "get_translation"]


def decode_and_evaluate(name,
                        model,
                        sess,
                        trans_file,
                        ref_file,
                        metrics,
                        bpe_delimiter,
                        beam_width,
                        tgt_eos,
                        decode=True):
  """Decode a test set and compute a score according to the evaluation task."""
  # Decode
  if decode:
    utils.print_out("  decoding to output %s." % trans_file)

    start_time = time.time()
    num_sentences = 0
    #with codecs.getwriter("utf-8")(
    #    tf.gfile.GFile(trans_file, mode="w")) as trans_f:
    #  trans_f.write("")  # Write empty string to ensure file is created.
    with tf.gfile.GFile(trans_file, mode="w") as trans_f:
      trans_f.write("")  # Write empty string to ensure file is created.      
      
      while True:
        try:
          nmt_outputs, _ = model.decode(sess)

          if beam_width > 0:
            # get the top translation.
            nmt_outputs = nmt_outputs[0]

          num_sentences += len(nmt_outputs)
          for sent_id in range(len(nmt_outputs)):
            translation = get_translation(
                nmt_outputs,
                sent_id,
                tgt_eos=tgt_eos,
                bpe_delimiter=bpe_delimiter)
            trans_f.write("%s\n" % translation)
        except tf.errors.OutOfRangeError:
          utils.print_time("  done, num sentences %d" % num_sentences,
                           start_time)
          break

  # Evaluation
  evaluation_scores = {}
  if ref_file and tf.gfile.Exists(trans_file):
    for metric in metrics:
      score = evaluation_utils.evaluate(
          ref_file,
          trans_file,
          metric,
          bpe_delimiter=bpe_delimiter)
      evaluation_scores[metric] = score
      utils.print_out("  %s %s: %.1f" % (metric, name, score))

  return evaluation_scores


def get_translation(nmt_outputs, sent_id, tgt_eos, bpe_delimiter):
  """Given batch decoding outputs, select a sentence and turn to text."""
  # Select a sentence
  output = nmt_outputs[sent_id, :].tolist()

  # If there is an eos symbol in outputs, cut them at that point.
  if tgt_eos and tgt_eos in output:
    output = output[:output.index(tgt_eos)]

  if not bpe_delimiter:
    translation = utils.format_text(output)
  else:  # BPE
    translation = utils.format_bpe_text(output, delimiter=bpe_delimiter)

  return translation


def get_logprobs(params, indices, batch_size=None, option_size=None):
  params = tf.expand_dims(params[:,0,:], axis=1)
  params = tf.transpose(tf.nn.log_softmax(params), perm=[1,0,2])

  if batch_size is None:
    batch_size = params.get_shape()[0].merge_with(indices.get_shape()[0]).value
    if batch_size is None:
      batch_size = tf.shape(indices)[0]
  if option_size is None:
    option_size = params.get_shape()[1].value
    if option_size is None:
      option_size = tf.shape(params)[1]

  batch_size_times_option_size = batch_size * option_size

  # gather_nd has no gradients implemented
  batch_time_params = tf.reshape(params, 
                                 tf.concat([[batch_size_times_option_size], 
                                            tf.shape(params)[2:]], axis=0))
  flat_params = tf.reshape(batch_time_params, [-1])
  indices_into_flat = tf.reshape(indices, [-1])
  flat_indices = tf.range(0, tf.shape(batch_time_params)[0]) *  tf.shape(batch_time_params)[1] + indices_into_flat

  log_probs =  tf.reshape(tf.gather(flat_params, flat_indices), [batch_size, -1])
  batch_log_probs = tf.reduce_sum(log_probs, axis=1)
    
  return batch_log_probs
