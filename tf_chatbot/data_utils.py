# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tensorflow as tf
from tensorflow.python.platform import gfile

from six.moves import urllib

# packages for Chinese process
import jieba
from snownlp import *


# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = [] 
# split by space
  for space_separated_fragment in sentence.strip().split():
# split by comma, question,...([.,!?\"':;)(])
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]


def basic_tokenizer_CN(sentence):
    s = SnowNLP(sentence)
    return s.words
   

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from %s" % (vocabulary_path, data_path))
        vocab = {}
        
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            
            for line in f:
                                
#                line = tf.compat.as_bytes(line)  # added by Ken
                counter += 1
                            
                if counter % 5000 == 0:
                    print("  processing line %d" % counter)
                       
                #tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                
                line = line.decode('utf-8')     #added by Tian 
                tokens = SnowNLP(line).words
                                
                for w in tokens:
                    
                    # replace numbers by all 0  
                    word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
                       
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            print('>> Full Vocabulary Size :',len(vocab_list))
            
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
                
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")
   
    else:
        print("Use existing vocabulary")

# vocab:  word-number
# reverse_vocab: number-word
def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            
            rev_vocab.extend(f.readlines())
            rev_vocab = [line.strip() for line in rev_vocab]
            
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
            
            return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):
    
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence) 
    
    words = SnowNLP(sentence).words
    
    # test 
    #print( len(words))
   
    if not normalize_digits:
        return [vocabulary.get(w.encode('utf-8'), UNK_ID) for w in words]
    
# Normalize digits by 0 before looking words up in the vocabulary.
# return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words] #mark added .decode by Ken
    
#   changed by Tian
    return [vocabulary.get(w.encode('utf-8'), UNK_ID) for w in words]
    
#    return [vocabulary.get(w.decode('utf-8'), UNK_ID) for w in words] # added by Ken


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):

  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="wb") as tokens_file:  # edit w to wb
        counter = 0
        for line in data_file:
            
            line = tf.compat.as_bytes(line)  # added by Ken
            line = line.decode('utf-8')     #added by tian
            
            counter += 1
            if counter % 5000 == 0:
                print("  tokenizing line %d" % counter)
                
            token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                              normalize_digits)
            tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n") 
  else:
    print("Already done")
    
#           test 
'''
            if counter ==17:
                print(line)
                
                words = SnowNLP(line).words
                
                for i in words:
                    print(i.encode('utf-8'))
                
                
                print(vocab.items()[15], type(vocab.keys()[15]), type(words[0]))
                
                
                print([vocab.get(w.encode('utf-8'), UNK_ID) for w in words])
                break
'''                  


def prepare_custom_data(working_directory, train_enc, train_dec, test_enc, test_dec, enc_vocabulary_size, dec_vocabulary_size, tokenizer=None):
    
    print("START\n")
    # Create vocabularies of the appropriate sizes.
    enc_vocab_path = os.path.join(working_directory, "vocab%d.enc" % enc_vocabulary_size)
    dec_vocab_path = os.path.join(working_directory, "vocab%d.dec" % dec_vocabulary_size)
    create_vocabulary(enc_vocab_path, train_enc, enc_vocabulary_size, tokenizer)
    create_vocabulary(dec_vocab_path, train_dec, dec_vocabulary_size, tokenizer)
 
    # Create token ids for the training data.
    enc_train_ids_path = train_enc + (".ids%d" % enc_vocabulary_size)
    dec_train_ids_path = train_dec + (".ids%d" % dec_vocabulary_size)
    data_to_token_ids(train_enc, enc_train_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(train_dec, dec_train_ids_path, dec_vocab_path, tokenizer)
    
    # Create token ids for the development data.
    enc_dev_ids_path = test_enc + (".ids%d" % enc_vocabulary_size)
    dec_dev_ids_path = test_dec + (".ids%d" % dec_vocabulary_size)
    data_to_token_ids(test_enc, enc_dev_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(test_dec, dec_dev_ids_path, dec_vocab_path, tokenizer)
    
    return (enc_train_ids_path, dec_train_ids_path, enc_dev_ids_path, dec_dev_ids_path, enc_vocab_path, dec_vocab_path)
    
