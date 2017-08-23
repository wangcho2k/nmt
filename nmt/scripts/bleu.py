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

"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..utils import misc_utils as utils

import collections
import math

import numpy as np

def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts

def _compute_bleu(reference_corpus,
                  translation_corpus,
                  max_order=4,
                  use_bp=True):
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  precisions = []

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams(references, max_order)
    translation_ngram_counts = _get_ngrams(translations, max_order)
   
    overlap = dict((ngram,
                    min(count, translation_ngram_counts[ngram]))
                   for ngram, count in ref_ngram_counts.items())

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram)-1] += translation_ngram_counts[ngram]

  precisions = [0] * max_order
  for i in xrange(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum/max_order)

  if use_bp:
    try:
      ratio = float(translation_length) / reference_length
      bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
    except:
      bp = 0.

  bleu = geo_mean * bp
  return np.float32(bleu)


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)

def _py_func(hypotheses, references, mrt_samples_meanloss=0):
  """
  Wrapper function that converts tensors to unicode and slices
  them until the EOS token is found.
  """
  def format_text(words):
    """
    Convert a sequence words into sentence.
    """
    if (not hasattr(words, "__len__") and  # for numpy array
      not isinstance(words, collections.Iterable)):
        words = [words]
    return b" ".join(words)
  
  def slice_text(text,
                 eos_token="</s>",
                 sos_token="<s>"):
    eos_index = text.find(eos_token)
    text = text[:eos_index] if eos_index > -1 else text
    sos_index = text.find(sos_token)
    text = text[sos_index+len(sos_token):] if sos_index > -1 else text
    return text.strip()

  hypotheses = np.array([format_text(_) for _ in hypotheses])
  references = np.array([format_text(_) for _ in references])
    
  # Deal with byte chars
  if hypotheses.dtype.kind == np.dtype("U"):
    hypotheses = np.char.encode(hypotheses, "utf-8")
  if references.dtype.kind == np.dtype("U"):
    references = np.char.encode(references, "utf-8")

  # Convert back to unicode object 
  hypotheses = [_.decode("utf-8") for _ in hypotheses]
  references = [_.decode("utf-8") for _ in references]

  # Slice all hypotheses and references up to SOS -> EOS
  sliced_hypotheses = [slice_text(_, '</s>', '<s>') for _ in hypotheses]
  sliced_references = [slice_text(_, '</s>', '<s>') for _ in references]
    
  """
  losses = np.zeros(len(sliced_hypotheses) + 1, dtype=np.float32)
  losses[0] = [1.]
  cnt = 1
  for hypothesis, reference in zip(sliced_hypotheses, sliced_references):
    metric_score = _compute_bleu([reference], [hypothesis])
    losses[cnt] = metric_score
    cnt += 1
  
  mean_loss = np.mean(0)
  if mrt_samples_meanloss > 0:
    mean_loss = np.mean(
        losses[0:mrt_samples_meanloss-1], dtype=np.float32)
   
  losses = mean_loss - losses
  utils.print_out(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  utils.print_out("%d" % len(losses))
  utils.print_out("%s" % losses)
  """
  
  losses = []
  for hypothesis, reference in zip(sliced_hypotheses, sliced_references):
    metric_score = _compute_bleu([reference], [hypothesis])
    losses.append(metric_score) 
  losses.insert(0, np.float32(1.0))
  
  mean_loss = np.mean(0.)
  if mrt_samples_meanloss > 0:
    mean_loss = np.mean(
        losses[0:mrt_samples_meanloss-1], dtype=np.float32)
   
  for i in xrange(len(losses)):
    losses[i] = mean_loss - losses[i]

  return losses
