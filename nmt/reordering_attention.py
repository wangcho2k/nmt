# Copyright 2017 Yongkeun Hwang. All Rights Reserved.
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

"""
A patched attention mechanisms for implementation of the distortion model on attention mechanism.

Reference :
Incorporating Word Reordering Knowledge into Attention-based Neural Machine Translation
J. Zhang et al., ACL 2017
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper

def _compute_attention_with_distortion(attention_mechanism, cell_output,
                                       previous_alignments, attention_layer,
                                       distorted_alignments):
    """
    Computes the attention and alignments for a given attention_mechanism
    with distorted alignments.
    """
    alignments = attention_mechanism(
      cell_output, previous_alignments=previous_alignments)
    with tf.control_dependencies([
        tf.assert_equal(tf.shape(alignments),tf.shape(distorted_alignments[-1,:,:]))]):
        alignments = 0.5*alignments + 0.5*distorted_alignments[-1,:,:]

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = array_ops.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, attention_mechanism.num_units]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, attention_mechanism.num_units].
    # we then squeeze out the singleton dim.
    context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
    context = array_ops.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(array_ops.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments, context

class ReorderingAttentionWrapper(attention_wrapper.AttentionWrapper):
    """
    Patched AttentionWrapper for distortion model
    """
    def __init__(self,
               cell,
               attention_mechanism,
               distortion_model,
               jump_distance,
               attention_layer_size=None,
               alignment_history=False,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               name=None):
        """Construct the `ReorderingAttentionWrapper`."""
        super(ReorderingAttentionWrapper, self).__init__(
            cell=cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=attention_layer_size,
            alignment_history=alignment_history,
            cell_input_fn=cell_input_fn,
            output_attention=output_attention,
            initial_cell_state=initial_cell_state,
            name=name)
        self.jump_distance = jump_distance

        # creating distortion model
        if distortion_model == "source":
            # S-distortion model, using previous attention as input
            pass
        elif distortion_model == "hidden":
            # H-distortion model, using previous cell state as input
            pass
        else:
            raise ValueError("Unknown distortion model %s" % distortion_model)
        self.dmodel = (layers_core.Dense(units=2 * jump_distance + 1, name="DistortionModel"),
                       distortion_model)

        # creating shifting matrix
        shifting_matrices = []
        max_time = self._attention_mechanisms[0].alignments_size
        for i in range(-1*jump_distance,0): # -k to -1
            tmp = tf.eye(max_time)
            tmp = tf.cond(abs(i) >= max_time,
                          lambda: 0 * tmp,
                          lambda: tf.concat([tf.zeros([-1*i,max_time]),tmp[0:i,:]],0))
            shifting_matrices.append(tmp)
        shifting_matrices.append(tf.eye(max_time)) # k = 0
        for i in range(1,jump_distance+1): # 1 to k
            tmp = tf.eye(max_time)
            tmp = tf.cond(i >= max_time,
                          lambda: 0 * tmp,
                          lambda: tf.concat([tmp[i:, :], tf.zeros([i, max_time])], 0))
            shifting_matrices.append(tmp)
        self.shifting_matrices = tf.stack(shifting_matrices)

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).
        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:
          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.
        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, attention_wrapper.AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_alignments = state.alignments
            previous_alignment_history = state.alignment_history
        else:
            previous_alignments = [state.alignments]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_histories = []
        all_contexts = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            if self.dmodel[1] == "source":
                distortion_output = self.dmodel[0](state.attention)
            elif self.dmodel[1] == "hidden":
                distortion_output = self.dmodel[0](state.cell_state[-1].h)
            distortion_output = tf.transpose(tf.nn.softmax(distortion_output)) # should be [7, batch]

            distorted_alignments = tf.scan(lambda a,t :
                                           a+tf.transpose(t[1]*tf.transpose(tf.matmul(previous_alignments[i],t[0]))),
                                           (self.shifting_matrices, distortion_output),
                                           initializer=tf.zeros_like(previous_alignments[i]))

            attention, alignments, context = _compute_attention_with_distortion(
                attention_mechanism, cell_output, previous_alignments[i],
                self._attention_layers[i] if self._attention_layers else None,
                distorted_alignments)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            #distorted_alignments = tf.Print(distorted_alignments,[distorted_alignments],summarize=40)

            all_alignments.append(alignments)
            all_histories.append(alignment_history)
            all_attentions.append(attention)
            all_contexts.append(context)

        attention = array_ops.concat(all_attentions, 1)
        context = array_ops.concat(all_contexts, 1)
        next_state = attention_wrapper.AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(all_histories))

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state


